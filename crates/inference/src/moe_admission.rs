//! Offline MoE expert-cache admission-policy simulator (issue #682, Stage 3:
//! "simulate admission policies against routing traces before touching the
//! engine's cache eviction logic").
//!
//! ## Why this exists
//!
//! [`crate::forward::moe_expert_cache`] ships a single admission/eviction
//! policy for the routed-expert dequant cache: bounded, per-(layer,
//! gate_up|down) LRU with a within-token "touched slot" eviction guard (see
//! that module's doc comment for the full GPU-buffer-lifetime rationale
//! this guard exists for). Published MoE-offloading literature (Mixtral-
//! Offloading, MoE-Infinity, AdapMoE, HOBBIT, ProMoE, ExpertFlow)
//! consistently reports that sequence-local-frequency or ARC-style
//! admission beats bare LRU on real routing traces. Before changing the
//! engine's eviction policy, this module lets a routing trace (JSONL, one
//! record per `(layer_idx, token_idx)`) be replayed offline against the
//! CURRENT policy and a small set of challenger policies, so any policy
//! change is justified by a measured hit-rate delta rather than intuition.
//! The pre-registered decision gate a challenger must clear on a real trace
//! (a >= 2.0 absolute-percentage-point overall hit-rate improvement over
//! the baseline LRU policy; ties keep LRU) is recorded alongside this
//! lane's synthetic-trace validation results, not in this module.
//!
//! This module is pure simulation: no Metal, no real weight dequant, no
//! change to [`crate::forward::moe_expert_cache`]'s runtime behavior. It
//! consumes only the expert *ids* a real forward pass selected, replaying
//! them against small in-memory policy structs.
//!
//! ## Trace format
//!
//! One JSON object per line, one line per `(layer_idx, token_idx)`:
//!
//! ```text
//! {"layer_idx": 3, "token_idx": 17, "selected_ids": [4, 91, 12, 200], "gate_weights": [0.31, 0.28, 0.22, 0.19]}
//! ```
//!
//! `gate_weights` is read but not used by any policy here — admission
//! decisions are id-sequence-only, matching what the engine's own LRU cache
//! observes. Extra fields are tolerated and ignored (flattened into a
//! discarded map) so this reader does not need to change in lockstep with
//! the trace-collector's schema.
//!
//! ## Fidelity notes
//!
//! [`LruPolicy`] mirrors [`crate::forward::moe_expert_cache::ExpertSlotCache`]'s
//! `touch`/`pick_eviction_slot` order exactly, including the within-token
//! "never evict a slot touched earlier this token" guard (moe_expert_cache.rs
//! `pick_eviction_slot`, `touch`, `plan_prefetch`). That guard protects
//! EVERY slot resolved earlier in the current token, not only literal
//! duplicate ids — production's `resolve` sets the touched flag on every
//! slot it touches, hit or miss, and `pick_eviction_slot` only ever selects
//! an untouched one (`moe_expert_cache.rs:47`-`58`, `887`-`912`). [`ArcPolicy`]
//! implements the equivalent token-local protection via
//! `protected_this_token` (see its doc comment). [`FreqAdmissionPolicy`]
//! needs no explicit guard — see its doc comment for the structural proof —
//! but both were audited against this invariant, not assumed safe because
//! `selected_ids` is duplicate-free (duplicate-freedom was never the
//! relevant property: evicting a *different* id touched earlier the same
//! token is just as physically unreachable as evicting a literal repeat).

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::BufRead;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Trace reading
// ---------------------------------------------------------------------------

/// One `(layer_idx, token_idx)` record from a routing trace.
#[derive(Debug, Clone, Deserialize)]
pub struct TraceRecord {
    pub layer_idx: usize,
    pub token_idx: usize,
    pub selected_ids: Vec<usize>,
    #[serde(default)]
    pub gate_weights: Vec<f32>,
    /// Tolerates any additional fields a newer trace-collector schema adds
    /// without breaking this reader — this reader targets exactly
    /// `layer_idx`/`token_idx`/`selected_ids`/`gate_weights` and must not
    /// hard-fail on extras from the sibling trace-collector lane.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A trace parse failure, naming the offending line.
#[derive(Debug)]
pub struct TraceReadError {
    pub line_no: usize,
    pub message: String,
}

impl std::fmt::Display for TraceReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line_no, self.message)
    }
}

impl std::error::Error for TraceReadError {}

/// Read a JSONL routing trace from any [`BufRead`] source. Blank lines are
/// skipped; anything else that fails to parse as a [`TraceRecord`] is a
/// hard error naming the offending line — a reader that silently dropped
/// malformed rows would corrupt the temporal LRU sequence for every layer,
/// not just the record itself.
pub fn read_trace<R: BufRead>(reader: R) -> Result<Vec<TraceRecord>, TraceReadError> {
    let mut records = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line_no = i + 1;
        let line = line.map_err(|e| TraceReadError {
            line_no,
            message: format!("I/O error: {e}"),
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: TraceRecord = serde_json::from_str(trimmed).map_err(|e| TraceReadError {
            line_no,
            message: format!("JSON parse error: {e}"),
        })?;
        records.push(record);
    }
    Ok(records)
}

/// Group trace records by `layer_idx`, sorting each layer's records by
/// `token_idx`. Required regardless of the trace file's own on-disk order —
/// a collector may legitimately emit token-major (every layer of token 0
/// before any of token 1) rather than layer-major — because every policy
/// below replays a per-layer cache strictly in token order.
pub fn group_by_layer(records: Vec<TraceRecord>) -> Vec<(usize, Vec<TraceRecord>)> {
    let mut by_layer: HashMap<usize, Vec<TraceRecord>> = HashMap::new();
    for r in records {
        by_layer.entry(r.layer_idx).or_default().push(r);
    }
    let mut layers: Vec<(usize, Vec<TraceRecord>)> = by_layer.into_iter().collect();
    for (_, recs) in layers.iter_mut() {
        recs.sort_by_key(|r| r.token_idx);
    }
    layers.sort_by_key(|(layer_idx, _)| *layer_idx);
    layers
}

// ---------------------------------------------------------------------------
// Admission policy trait
// ---------------------------------------------------------------------------

/// One access outcome the simulator's driver tallies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOutcome {
    Hit,
    Miss,
}

/// A pluggable expert-cache admission/eviction policy, replayed one
/// `(layer, token)` at a time. Each implementation models a single
/// (layer, gate_up|down)-shaped cache of fixed `capacity` slots — see each
/// impl's doc comment for its eviction rule. Object-safe so the simulation
/// driver can hold policies as `Box<dyn AdmissionPolicy>`.
pub trait AdmissionPolicy {
    /// Human-readable policy name for reporting.
    fn name(&self) -> &'static str;
    /// Called once per token, before that token's `access` calls. Clears
    /// per-token bookkeeping; a no-op for policies that don't need the
    /// same-token eviction guard (see their doc comments).
    fn begin_token(&mut self);
    /// Resolve one expert id, returning whether it was already resident.
    fn access(&mut self, expert_id: usize) -> AccessOutcome;
    /// Cumulative count of resident-data evictions (ARC's ghost-list
    /// pruning does not count — no resident data is discarded by it).
    fn evictions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Baseline: faithful LRU mirror
// ---------------------------------------------------------------------------

/// Faithful mirror of
/// [`crate::forward::moe_expert_cache::ExpertSlotCache`]'s admission and
/// eviction order (Stage 1, shipped): a fixed pool of `capacity` slots,
/// each holding at most one expert id, evicted least-recently-used, with
/// the "never evict a slot touched earlier THIS token" guard from that
/// module's `pick_eviction_slot`/`touch` (moe_expert_cache.rs, roughly
/// lines 887-969) — required there because the real cache's slots back
/// live Metal buffers a single token's still-open command buffer may
/// reference (see that module's doc comment for the full invariant).
///
/// Driving this policy with `access()` once per id in trace order
/// reproduces exactly the same side effects as a `plan_prefetch` call over
/// the same ids in the same order — `moe_expert_cache.rs`'s own doc
/// comment for `plan_prefetch` states this equivalence explicitly — so
/// this simulator does not need to separately model the real cache's
/// plan/spawn/apply split.
pub struct LruPolicy {
    slot_owner: Vec<Option<usize>>,
    expert_to_slot: HashMap<usize, usize>,
    slot_touched: Vec<bool>,
    /// Slot indices in LRU order, front = least-recently-used.
    lru: VecDeque<usize>,
    evictions: usize,
}

impl LruPolicy {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LruPolicy requires capacity > 0");
        Self {
            slot_owner: vec![None; capacity],
            expert_to_slot: HashMap::with_capacity(capacity),
            slot_touched: vec![false; capacity],
            lru: (0..capacity).collect(),
            evictions: 0,
        }
    }

    fn touch(&mut self, slot: usize) {
        self.slot_touched[slot] = true;
        if let Some(pos) = self.lru.iter().position(|&s| s == slot) {
            self.lru.remove(pos);
        }
        self.lru.push_back(slot);
    }

    fn pick_eviction_slot(&mut self) -> usize {
        let pos = self.lru.iter().position(|&s| !self.slot_touched[s]).expect(
            "no untouched slot available for eviction — capacity must be >= the number of \
                 distinct experts one token requests (mirrors moe_expert_cache_num_slots's \
                 num_slots >= top_k invariant); this indicates a misconfigured \
                 --num-slots/--top-k pair, not a trace problem",
        );
        self.lru
            .remove(pos)
            .expect("pos was just found via position() on this same self.lru")
    }
}

impl AdmissionPolicy for LruPolicy {
    fn name(&self) -> &'static str {
        "lru (baseline)"
    }

    fn begin_token(&mut self) {
        self.slot_touched.iter_mut().for_each(|t| *t = false);
    }

    fn access(&mut self, expert_id: usize) -> AccessOutcome {
        if let Some(&slot) = self.expert_to_slot.get(&expert_id) {
            self.touch(slot);
            return AccessOutcome::Hit;
        }
        let slot = self.pick_eviction_slot();
        if let Some(old_owner) = self.slot_owner[slot].take() {
            self.expert_to_slot.remove(&old_owner);
            self.evictions += 1;
        }
        self.slot_owner[slot] = Some(expert_id);
        self.expert_to_slot.insert(expert_id, slot);
        self.touch(slot);
        AccessOutcome::Miss
    }

    fn evictions(&self) -> usize {
        self.evictions
    }
}

// ---------------------------------------------------------------------------
// Challenger 1: ARC (Adaptive Replacement Cache)
// ---------------------------------------------------------------------------

/// Adaptive Replacement Cache (Megiddo & Modha, FAST 2003), adapted to
/// expert ids as the cached "pages": `t1`/`t2` are resident recency-/
/// frequency-segmented lists (`t1.len() + t2.len() <= capacity`), `b1`/`b2`
/// are ghost (id-only, no data) histories of ids recently evicted from
/// `t1`/`t2`, and `p` is the adaptively-tuned target size for `t1`. A ghost
/// hit (`b1`/`b2`) nudges `p` toward whichever list is proving more
/// valuable, then re-admits the id directly into `t2` (frequency segment)
/// without needing a third touch — the mechanism the MoE-offloading
/// literature credits for beating bare LRU under bursty expert reuse
/// (e.g. a small hot expert set interleaved with a long one-off scan).
///
/// Same-token eviction guard, analogous to [`LruPolicy`]'s: `access`
/// records every id it resolves (hit or miss) into `protected_this_token`,
/// cleared each `begin_token`. Eviction victim selection (in `replace` and
/// Case IV's direct `t1` discard) skips protected ids — a token resolved
/// earlier THIS token must not be evicted before the token boundary,
/// exactly as production requires (see the module doc comment and
/// `moe_expert_cache.rs:47`-`58`, `887`-`912`). With `capacity >= top_k`
/// (validated by `run_simulation`), an unprotected resident always exists
/// when eviction is needed: a token touches at most `top_k` distinct ids,
/// so at most `top_k - 1` residents can be protected by the time the next
/// one needs room. ARC's `p`-adaptive target still picks WHICH list
/// (`t1`/`t2`) to prefer evicting from; the guard only constrains WHICH
/// resident within that choice, falling back to the other list if the
/// preferred one is fully protected — the physical slot has to come from
/// somewhere, and which ghost list absorbs it is a secondary adaptation
/// detail, not a correctness requirement.
pub struct ArcPolicy {
    capacity: usize,
    p: usize,
    t1: VecDeque<usize>,
    t2: VecDeque<usize>,
    b1: VecDeque<usize>,
    b2: VecDeque<usize>,
    evictions: usize,
    /// Ids resolved earlier in the current token — see the struct doc
    /// comment. Cleared by `begin_token`.
    protected_this_token: HashSet<usize>,
}

impl ArcPolicy {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ArcPolicy requires capacity > 0");
        Self {
            capacity,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            evictions: 0,
            protected_this_token: HashSet::new(),
        }
    }

    /// Resident id count across `t1` + `t2` — must never exceed `capacity`;
    /// exercised by the invariant test below.
    #[cfg(test)]
    fn resident_len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    fn remove_from(list: &mut VecDeque<usize>, id: usize) -> bool {
        if let Some(pos) = list.iter().position(|&x| x == id) {
            list.remove(pos);
            true
        } else {
            false
        }
    }

    /// Remove and return the oldest (front-most) id in `list` that is NOT
    /// in `protected` — the LRU-among-unprotected victim. `None` if every
    /// id in `list` is protected this token.
    fn evict_unprotected_from(
        list: &mut VecDeque<usize>,
        protected: &HashSet<usize>,
    ) -> Option<usize> {
        let pos = list.iter().position(|id| !protected.contains(id))?;
        list.remove(pos)
    }

    /// REPLACE(x, p): evict one resident id from `t1` or `t2` into the
    /// corresponding ghost list, per the ARC paper's rule — prefer
    /// evicting from `t1` when it exceeds its adaptive target `p` (or sits
    /// exactly at `p` and the id that triggered this REPLACE call came
    /// from `b2`), otherwise evict from `t2`. The victim is additionally
    /// constrained to ids not in `protected_this_token` (see the struct
    /// doc comment); if the ARC-preferred list has no unprotected
    /// candidate, fall back to the other resident list before giving up.
    fn replace(&mut self, triggered_by_b2_hit: bool) {
        let evict_from_t1 = !self.t1.is_empty()
            && (self.t1.len() > self.p || (triggered_by_b2_hit && self.t1.len() == self.p));
        let victim = if evict_from_t1 {
            Self::evict_unprotected_from(&mut self.t1, &self.protected_this_token)
                .map(|y| (y, true))
                .or_else(|| {
                    Self::evict_unprotected_from(&mut self.t2, &self.protected_this_token)
                        .map(|y| (y, false))
                })
        } else {
            Self::evict_unprotected_from(&mut self.t2, &self.protected_this_token)
                .map(|y| (y, false))
                .or_else(|| {
                    Self::evict_unprotected_from(&mut self.t1, &self.protected_this_token)
                        .map(|y| (y, true))
                })
        };
        match victim {
            Some((y, from_t1)) => {
                self.b2_or_b1_push(from_t1, y);
                self.evictions += 1;
            }
            None => panic!(
                "ArcPolicy::replace: no unprotected resident available for eviction — \
                 indicates capacity < top_k (should have been rejected by \
                 run_simulation) or a bug in protected_this_token bookkeeping"
            ),
        }
    }

    fn b2_or_b1_push(&mut self, from_t1: bool, id: usize) {
        if from_t1 {
            self.b1.push_back(id);
        } else {
            self.b2.push_back(id);
        }
    }
}

impl AdmissionPolicy for ArcPolicy {
    fn name(&self) -> &'static str {
        "arc"
    }

    fn begin_token(&mut self) {
        self.protected_this_token.clear();
    }

    fn access(&mut self, x: usize) -> AccessOutcome {
        let c = self.capacity;
        // Mark x protected from eviction by any LATER access in this same
        // token, mirroring production's `resolve` setting the touched
        // flag on every slot it touches, hit or miss (see struct doc
        // comment). x itself is never a candidate for eviction within
        // this call — it is not yet resident in t1/t2 at this point on
        // any path that evicts something to make room for it.
        self.protected_this_token.insert(x);

        // Case I: cache hit.
        if Self::remove_from(&mut self.t1, x) {
            self.t2.push_back(x);
            return AccessOutcome::Hit;
        }
        if self.t2.contains(&x) {
            Self::remove_from(&mut self.t2, x);
            self.t2.push_back(x);
            return AccessOutcome::Hit;
        }

        // Case II: ghost hit in b1 — recency history says "admit sooner".
        // Ratio uses |b1| INCLUDING x (matches the ARC paper's ordering:
        // adapt p, then REPLACE, then move x out of b1) — computed before
        // the removal below, not after.
        if self.b1.contains(&x) {
            let ratio = (self.b2.len() / self.b1.len().max(1)).max(1);
            self.p = (self.p + ratio).min(c);
            self.replace(false);
            Self::remove_from(&mut self.b1, x);
            self.t2.push_back(x);
            return AccessOutcome::Miss;
        }

        // Case III: ghost hit in b2 — frequency history says "admit later".
        // Same before-removal ratio ordering as Case II above.
        if self.b2.contains(&x) {
            let ratio = (self.b1.len() / self.b2.len().max(1)).max(1);
            self.p = self.p.saturating_sub(ratio);
            self.replace(true);
            Self::remove_from(&mut self.b2, x);
            self.t2.push_back(x);
            return AccessOutcome::Miss;
        }

        // Case IV: total miss — not resident, not in either ghost list.
        let t1_b1 = self.t1.len() + self.b1.len();
        if t1_b1 == c {
            if self.t1.len() < c {
                self.b1.pop_front();
                self.replace(false);
            } else {
                match Self::evict_unprotected_from(&mut self.t1, &self.protected_this_token) {
                    Some(_) => self.evictions += 1,
                    None => panic!(
                        "ArcPolicy Case IV: t1 full at capacity with b1 empty, but every \
                         resident id is protected this token — indicates capacity < \
                         top_k (should have been rejected by run_simulation)"
                    ),
                }
            }
        } else if t1_b1 < c {
            let total = self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len();
            if total >= c {
                if total == 2 * c {
                    self.b2.pop_front();
                }
                self.replace(false);
            }
        }
        self.t1.push_back(x);
        AccessOutcome::Miss
    }

    fn evictions(&self) -> usize {
        self.evictions
    }
}

// ---------------------------------------------------------------------------
// Challenger 2: sequence-local frequency admission
// ---------------------------------------------------------------------------

/// Sequence-local frequency admission: an id is only ADMITTED into the
/// resident cache once it has been seen at least twice within a sliding
/// window of `window` accesses (to this same per-layer policy instance).
/// The first sighting of an id is always a miss with no cache mutation
/// beyond recording when it was seen. Once admitted, eviction among
/// resident ids is plain LRU. This models the cheapest form of "don't
/// cache one-shot experts" admission filtering the MoE-offloading
/// literature (e.g. HOBBIT, ProMoE) motivates as a low-overhead
/// alternative to full ARC.
///
/// No explicit same-token eviction guard is needed, unlike [`ArcPolicy`].
/// `admit`'s only eviction path is `self.lru.pop_front()`, and every
/// successful `access` (hit or admission) moves that id to the back of
/// `self.lru` before returning — so an id touched earlier in the current
/// token is always more-recently-touched than any not-yet-touched id, and
/// can only become the front-of-queue victim if it is the cache's ONLY
/// resident (`capacity == 1`). A token with more than one distinct
/// selected id requires `capacity > 1` (`run_simulation` validates
/// `capacity (num_slots) >= top_k`), so that degenerate case cannot occur
/// for a multi-id token. See `freq_admission_never_evicts_a_same_token_touched_id_without_an_explicit_guard`
/// for the proof trace.
pub struct FreqAdmissionPolicy {
    capacity: usize,
    window: usize,
    resident: HashSet<usize>,
    lru: VecDeque<usize>,
    last_seen_pos: HashMap<usize, usize>,
    pos: usize,
    evictions: usize,
}

impl FreqAdmissionPolicy {
    pub fn new(capacity: usize, window: usize) -> Self {
        assert!(capacity > 0, "FreqAdmissionPolicy requires capacity > 0");
        assert!(window > 0, "FreqAdmissionPolicy requires window > 0");
        Self {
            capacity,
            window,
            resident: HashSet::with_capacity(capacity),
            lru: VecDeque::new(),
            last_seen_pos: HashMap::new(),
            pos: 0,
            evictions: 0,
        }
    }

    fn touch_lru(&mut self, id: usize) {
        if let Some(p) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(p);
        }
        self.lru.push_back(id);
    }

    fn admit(&mut self, id: usize) {
        if self.resident.len() >= self.capacity
            && let Some(victim) = self.lru.pop_front()
        {
            self.resident.remove(&victim);
            self.evictions += 1;
        }
        self.resident.insert(id);
        self.lru.push_back(id);
    }
}

impl AdmissionPolicy for FreqAdmissionPolicy {
    fn name(&self) -> &'static str {
        "freq-admission"
    }

    fn begin_token(&mut self) {}

    fn access(&mut self, id: usize) -> AccessOutcome {
        self.pos += 1;
        if self.resident.contains(&id) {
            self.touch_lru(id);
            return AccessOutcome::Hit;
        }
        // Strict `<`, not `<=`: a `window`-access sliding window ending at
        // the current position `pos` covers positions
        // `[pos - window + 1, pos]`, i.e. `pos - prev < window`. A prior
        // touch exactly `window` positions back sits one position outside
        // that range.
        let eligible = self
            .last_seen_pos
            .get(&id)
            .is_some_and(|&prev| self.pos - prev < self.window);
        self.last_seen_pos.insert(id, self.pos);
        if eligible {
            self.admit(id);
        }
        AccessOutcome::Miss
    }

    fn evictions(&self) -> usize {
        self.evictions
    }
}

// ---------------------------------------------------------------------------
// Policy catalog + simulation driver
// ---------------------------------------------------------------------------

/// Named policy constructor — how the driver enumerates the baseline plus
/// challenger policies (issue #682 Stage 3 item 3), each rebuilt fresh per
/// layer — no cross-layer cache sharing. This models ONE logical
/// (layer, tensor) cache per layer, NOT the two physical
/// `ExpertSlotCache` instances (gate_up and down) production actually
/// runs per MoE layer (`metal_qwen35.rs` constructs both with the same
/// `num_slots` and plans both against the same `selected` id sequence —
/// see [`LayerStats`]'s doc comment for what this means for reported raw
/// counts). Adding a new challenger is one new variant plus one
/// `AdmissionPolicy` impl.
#[derive(Debug, Clone)]
pub enum PolicySpec {
    Lru,
    Arc,
    FreqAdmission { window: usize },
}

impl PolicySpec {
    pub fn label(&self) -> String {
        match self {
            PolicySpec::Lru => "lru (baseline)".to_string(),
            PolicySpec::Arc => "arc".to_string(),
            PolicySpec::FreqAdmission { window } => format!("freq-admission(w={window})"),
        }
    }

    pub fn build(&self, capacity: usize) -> Box<dyn AdmissionPolicy> {
        match self {
            PolicySpec::Lru => Box::new(LruPolicy::new(capacity)),
            PolicySpec::Arc => Box::new(ArcPolicy::new(capacity)),
            PolicySpec::FreqAdmission { window } => {
                Box::new(FreqAdmissionPolicy::new(capacity, *window))
            }
        }
    }

    pub fn is_baseline(&self) -> bool {
        matches!(self, PolicySpec::Lru)
    }
}

/// Hit/miss/eviction tally for one policy over one layer's (or the whole
/// trace's) accesses.
///
/// **These counts are per single LOGICAL (layer, tensor) cache, not
/// physical runtime events.** Production instantiates two `ExpertSlotCache`
/// instances per MoE layer (gate_up and down) and replays the same
/// selected-expert id sequence into both (`metal_qwen35.rs`'s
/// `encode_moe_ffn`), so each miss and eviction reported here happens
/// TWICE physically — once per tensor — with identical resident histories
/// in both (same ids, same order, same `num_slots`). `hit_rate` and
/// `delta_vs_baseline_pp` are unaffected by this (both tensors compute the
/// same ratio); only the raw `hits`/`misses`/`evictions` integers
/// understate physical dequant/eviction event counts by 2x. Multiply by 2
/// for physical totals, or treat these as the logical/representative
/// count they are.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LayerStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl LayerStats {
    pub fn total(&self) -> usize {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        if self.total() == 0 {
            0.0
        } else {
            self.hits as f64 / self.total() as f64
        }
    }
}

/// Run one policy instance over one layer's token-ordered records, calling
/// `begin_token` once per record before that token's `access` calls — the
/// same per-token contract [`crate::forward::moe_expert_cache::ExpertSlotCache`]
/// requires of its own callers.
pub fn simulate_layer(records: &[TraceRecord], policy: &mut dyn AdmissionPolicy) -> LayerStats {
    let mut stats = LayerStats::default();
    for record in records {
        policy.begin_token();
        for &id in &record.selected_ids {
            match policy.access(id) {
                AccessOutcome::Hit => stats.hits += 1,
                AccessOutcome::Miss => stats.misses += 1,
            }
        }
    }
    stats.evictions = policy.evictions();
    stats
}

/// Cache-sizing knobs for a simulation run — the CLI's `--num-slots`/
/// `--top-k` inputs.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub num_slots: usize,
    pub top_k: usize,
}

/// One layer's stats row in a [`PolicyReport`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LayerRow {
    pub layer_idx: usize,
    pub stats: LayerStats,
}

/// Full simulation result for one policy: an overall (all layers combined)
/// tally, a per-layer breakdown, and the overall hit-rate delta (in
/// percentage points) versus the baseline LRU policy.
#[derive(Debug, Clone, Serialize)]
pub struct PolicyReport {
    pub policy: String,
    pub overall: LayerStats,
    pub per_layer: Vec<LayerRow>,
    pub delta_vs_baseline_pp: f64,
}

/// Replay `layers` (as produced by [`group_by_layer`]) against every policy
/// in `policies`, returning one [`PolicyReport`] per policy. `policies`
/// MUST include [`PolicySpec::Lru`] — it is the baseline every other
/// policy's `delta_vs_baseline_pp` is measured against.
pub fn run_simulation(
    layers: &[(usize, Vec<TraceRecord>)],
    cfg: &SimConfig,
    policies: &[PolicySpec],
) -> Result<Vec<PolicyReport>, String> {
    // Fail closed on zero-valued sizing inputs BEFORE any policy is
    // constructed — `LruPolicy::new`/`ArcPolicy::new`/
    // `FreqAdmissionPolicy::new` assert `capacity/window > 0` and will
    // panic on these, which is fine as an internal invariant but wrong as
    // the response to an ordinary bad CLI argument (e.g. `--num-slots 0`
    // or `--window 0`). Checked here, not just in the CLI, so it is
    // covered by `cargo test -p lattice-inference --lib` directly.
    if cfg.num_slots == 0 {
        return Err(
            "num_slots must be > 0 — a zero-capacity cache cannot hold any \
                     expert"
                .to_string(),
        );
    }
    if cfg.top_k == 0 {
        return Err(
            "top_k must be > 0 — a token that selects zero experts is not a \
                     valid MoE routing trace"
                .to_string(),
        );
    }
    if cfg.num_slots < cfg.top_k {
        return Err(format!(
            "num_slots ({}) must be >= top_k ({}) — mirrors moe_expert_cache_num_slots's \
             validated invariant; a smaller cache cannot serve one token's routed-expert set",
            cfg.num_slots, cfg.top_k
        ));
    }
    for spec in policies {
        if let PolicySpec::FreqAdmission { window } = spec
            && *window == 0
        {
            return Err(
                "freq-admission window must be > 0 — a zero-width sliding window can \
                 never observe a repeat touch"
                    .to_string(),
            );
        }
    }
    if !policies.iter().any(PolicySpec::is_baseline) {
        return Err("policies must include PolicySpec::Lru as the baseline".to_string());
    }

    for (layer_idx, records) in layers {
        for r in records {
            if r.selected_ids.len() > cfg.top_k {
                return Err(format!(
                    "layer {layer_idx} token {}: selected_ids has {} entries, exceeds \
                     --top-k={}",
                    r.token_idx,
                    r.selected_ids.len(),
                    cfg.top_k
                ));
            }
        }
    }

    let mut reports = Vec::with_capacity(policies.len());
    let mut baseline_hit_rate = 0.0;

    for spec in policies {
        let mut overall = LayerStats::default();
        let mut per_layer = Vec::with_capacity(layers.len());
        for (layer_idx, records) in layers {
            let mut policy = spec.build(cfg.num_slots);
            let stats = simulate_layer(records, policy.as_mut());
            overall.hits += stats.hits;
            overall.misses += stats.misses;
            overall.evictions += stats.evictions;
            per_layer.push(LayerRow {
                layer_idx: *layer_idx,
                stats,
            });
        }
        if spec.is_baseline() {
            baseline_hit_rate = overall.hit_rate();
        }
        reports.push(PolicyReport {
            policy: spec.label(),
            overall,
            per_layer,
            delta_vs_baseline_pp: 0.0,
        });
    }

    for report in reports.iter_mut() {
        report.delta_vs_baseline_pp = (report.overall.hit_rate() - baseline_hit_rate) * 100.0;
    }

    Ok(reports)
}

/// Render `reports` as a plain-text table: an overall summary row per
/// policy, then a per-layer breakdown.
pub fn format_table(reports: &[PolicyReport]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "NOTE: hits/misses/evictions are per single LOGICAL (layer, tensor) cache. \
         Production runs TWO such caches per MoE layer (gate_up + down) replaying the \
         same selected-expert sequence into both — physical event totals are these \
         figures x2; hit_rate/delta_pp are unaffected (see LayerStats doc comment)."
    );
    let _ = writeln!(out, "=== overall (all layers) ===");
    let _ = writeln!(
        out,
        "{:<24} {:>10} {:>8} {:>8} {:>10} {:>10}",
        "policy", "hit_rate", "hits", "misses", "evictions", "delta_pp"
    );
    for r in reports {
        let _ = writeln!(
            out,
            "{:<24} {:>9.2}% {:>8} {:>8} {:>10} {:>+10.2}",
            r.policy,
            r.overall.hit_rate() * 100.0,
            r.overall.hits,
            r.overall.misses,
            r.overall.evictions,
            r.delta_vs_baseline_pp,
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "=== per-layer ===");
    let _ = writeln!(
        out,
        "{:<24} {:>8} {:>10} {:>8} {:>8} {:>10}",
        "policy", "layer", "hit_rate", "hits", "misses", "evictions"
    );
    for r in reports {
        for row in &r.per_layer {
            let _ = writeln!(
                out,
                "{:<24} {:>8} {:>9.2}% {:>8} {:>8} {:>10}",
                r.policy,
                row.layer_idx,
                row.stats.hit_rate() * 100.0,
                row.stats.hits,
                row.stats.misses,
                row.stats.evictions,
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(layer_idx: usize, token_idx: usize, selected_ids: &[usize]) -> TraceRecord {
        TraceRecord {
            layer_idx,
            token_idx,
            selected_ids: selected_ids.to_vec(),
            gate_weights: Vec::new(),
            extra: HashMap::new(),
        }
    }

    /// `VecDeque` has no `PartialEq<Vec<_>>` impl — convert for `assert_eq!`.
    fn dq(v: &VecDeque<usize>) -> Vec<usize> {
        v.iter().copied().collect()
    }

    // -----------------------------------------------------------------
    // Trace reading
    // -----------------------------------------------------------------

    #[test]
    fn reads_jsonl_and_tolerates_extra_fields() {
        let jsonl = "\
{\"layer_idx\": 0, \"token_idx\": 0, \"selected_ids\": [1, 2], \"gate_weights\": [0.5, 0.5]}

{\"layer_idx\": 1, \"token_idx\": 0, \"selected_ids\": [3], \"gate_weights\": [1.0], \"extra_field\": \"ignored\", \"nested\": {\"a\": 1}}
";
        let records =
            read_trace(jsonl.as_bytes()).expect("valid trace parses (blank line skipped)");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].layer_idx, 0);
        assert_eq!(records[0].selected_ids, vec![1, 2]);
        assert_eq!(records[1].layer_idx, 1);
        assert_eq!(records[1].selected_ids, vec![3]);
        assert!(records[1].extra.contains_key("extra_field"));
        assert!(records[1].extra.contains_key("nested"));
    }

    #[test]
    fn malformed_line_errors_with_line_number() {
        let jsonl = "{\"layer_idx\": 0, \"token_idx\": 0, \"selected_ids\": [1]}\nnot json\n";
        let err = read_trace(jsonl.as_bytes()).expect_err("malformed second line must error");
        assert_eq!(err.line_no, 2);
    }

    #[test]
    fn group_by_layer_sorts_within_layer_regardless_of_file_order() {
        // Token-major on disk (layer interleaved) — grouping must still
        // recover strict per-layer token order.
        let records = vec![
            record(0, 2, &[1]),
            record(1, 0, &[9]),
            record(0, 0, &[1]),
            record(1, 1, &[9]),
            record(0, 1, &[1]),
        ];
        let layers = group_by_layer(records);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].0, 0);
        let token_order: Vec<usize> = layers[0].1.iter().map(|r| r.token_idx).collect();
        assert_eq!(token_order, vec![0, 1, 2]);
        assert_eq!(layers[1].0, 1);
        let token_order: Vec<usize> = layers[1].1.iter().map(|r| r.token_idx).collect();
        assert_eq!(token_order, vec![0, 1]);
    }

    // -----------------------------------------------------------------
    // 5(c): hand-computed LRU sequence, including the within-token
    // touched-slot eviction guard (the exact invariant
    // moe_expert_cache.rs's pick_eviction_slot/touch enforce).
    // -----------------------------------------------------------------

    #[test]
    fn lru_hand_computed_hit_miss_sequence_and_within_token_touch_guard() {
        let mut policy = LruPolicy::new(2);
        let mut outcomes = Vec::new();

        // token 0: A, B — both cold (cache starts empty).
        policy.begin_token();
        outcomes.push(policy.access(100)); // A
        outcomes.push(policy.access(200)); // B

        // token 1: A (hit), C (miss — evicts B, the untouched slot this
        // token; A is protected because it was just touched this token).
        policy.begin_token();
        outcomes.push(policy.access(100)); // A
        outcomes.push(policy.access(300)); // C

        // token 2: B (miss — evicts A; A's "touched" protection from
        // token 1 must NOT carry over, proving begin_token resets it per
        // token), C (hit).
        policy.begin_token();
        outcomes.push(policy.access(200)); // B
        outcomes.push(policy.access(300)); // C

        use AccessOutcome::{Hit, Miss};
        assert_eq!(
            outcomes,
            vec![Miss, Miss, Hit, Miss, Miss, Hit],
            "hand-computed LRU hit/miss sequence mismatch"
        );
        assert_eq!(
            policy.evictions(),
            2,
            "expected exactly 2 evictions (B at token 1, A at token 2)"
        );
    }

    /// Same scenario, driven through the full `simulate_layer` +
    /// `TraceRecord` path (not the policy directly) — proves the driver's
    /// per-record `begin_token` wiring matches the hand trace above.
    #[test]
    fn simulate_layer_reproduces_hand_computed_lru_sequence() {
        let records = vec![
            record(0, 0, &[100, 200]),
            record(0, 1, &[100, 300]),
            record(0, 2, &[200, 300]),
        ];
        let mut policy = LruPolicy::new(2);
        let stats = simulate_layer(&records, &mut policy);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 4);
        assert_eq!(stats.evictions, 2);
        assert!((stats.hit_rate() - (2.0 / 6.0)).abs() < 1e-9);
    }

    // -----------------------------------------------------------------
    // 5(b): pure-LRU-friendly cyclic trace — ARC and LRU tie.
    // -----------------------------------------------------------------

    #[test]
    fn arc_ties_lru_on_cache_resident_cyclic_pattern() {
        let capacity = 4;
        let records: Vec<TraceRecord> = (0..50)
            .map(|token_idx| record(0, token_idx, &[token_idx % capacity]))
            .collect();

        let mut lru = LruPolicy::new(capacity);
        let lru_stats = simulate_layer(&records, &mut lru);

        let mut arc = ArcPolicy::new(capacity);
        let arc_stats = simulate_layer(&records, &mut arc);

        // Working set (4 distinct ids) exactly fits capacity (4): after the
        // first full cycle, neither policy should ever evict again, so
        // both tie exactly.
        assert_eq!(lru_stats.evictions, 0);
        assert_eq!(arc_stats.evictions, 0);
        assert_eq!(lru_stats.hits, arc_stats.hits);
        assert_eq!(lru_stats.misses, arc_stats.misses);
        assert!((lru_stats.hit_rate() - arc_stats.hit_rate()).abs() < 1e-9);
        // Sanity: this really is a high-hit-rate regime, not a
        // vacuous all-miss tie.
        assert!(lru_stats.hit_rate() > 0.8, "got {}", lru_stats.hit_rate());
    }

    // -----------------------------------------------------------------
    // 5(a): repeat-heavy trace (hot set + polluting scan) — ARC beats LRU.
    // -----------------------------------------------------------------

    #[test]
    fn arc_beats_lru_on_hot_set_plus_scan_pollution() {
        let capacity = 4;
        let hot_set = [0usize, 1usize];
        let scan_size = 8;
        let rounds = 30;

        // Each round: touch the 2-id hot set TWICE each (back to back),
        // then flood with `scan_size` never-repeated ids — a one-off scan
        // long enough to fully displace a bare-LRU cache of `capacity`
        // slots before the next round's hot-set accesses arrive. The
        // back-to-back double touch matters: a single touch per round
        // never gives ARC's `t1` a chance to promote a hot id into `t2`
        // (frequency-protected) before the very next round's scan evicts
        // it straight out of `t1` — capacity(4) is smaller than one
        // round's non-hot traffic (scan_size=8), so a once-touched id
        // never survives to be re-touched a round later. The immediate
        // repeat is what lets ARC's Case I promotion (`t1` hit -> moved
        // to `t2`) run at all; without it this test would degenerate to
        // ARC behaving identically to bare LRU (both purely FIFO-evicting
        // out of `t1`, no ghost-list learning ever triggered) and the
        // assertion below would fail. top_k=1 for this synthetic trace
        // (one id selected per token).
        let mut records = Vec::new();
        let mut token_idx = 0;
        let mut next_scan_id = 1000usize;
        for _round in 0..rounds {
            for &hot_id in &hot_set {
                records.push(record(0, token_idx, &[hot_id]));
                token_idx += 1;
            }
            for &hot_id in &hot_set {
                records.push(record(0, token_idx, &[hot_id]));
                token_idx += 1;
            }
            for _ in 0..scan_size {
                records.push(record(0, token_idx, &[next_scan_id]));
                next_scan_id += 1;
                token_idx += 1;
            }
        }

        let mut lru = LruPolicy::new(capacity);
        let lru_stats = simulate_layer(&records, &mut lru);

        let mut arc = ArcPolicy::new(capacity);
        let arc_stats = simulate_layer(&records, &mut arc);

        assert!(
            arc_stats.hit_rate() > lru_stats.hit_rate() + 0.02,
            "expected ARC to beat LRU by >2pp on hot-set+scan trace: lru={:.4} arc={:.4}",
            lru_stats.hit_rate(),
            arc_stats.hit_rate()
        );
    }

    #[test]
    fn arc_never_exceeds_capacity() {
        let capacity = 3;
        let mut arc = ArcPolicy::new(capacity);
        // Mixed pattern: repeats, one-offs, and ghost-list churn. One id
        // per token (begin_token before each) — isolates ARC's ordinary
        // per-request adaptation from the same-token protection guard
        // exercised separately below.
        let ids = [
            1, 2, 3, 4, 1, 5, 2, 6, 7, 1, 8, 9, 2, 10, 1, 2, 3, 11, 12, 13,
        ];
        for &id in &ids {
            arc.begin_token();
            arc.access(id);
            assert!(
                arc.resident_len() <= capacity,
                "resident count exceeded capacity after accessing {id}"
            );
        }
    }

    // -----------------------------------------------------------------
    // ArcPolicy must not evict an id resolved earlier in the SAME token.
    // Replays the exact three-token scenario that regressed this invariant.
    // -----------------------------------------------------------------

    #[test]
    fn arc_never_evicts_a_same_token_touched_id() {
        // capacity = top_k = 2. Token 0 fills t1 = [X, Y] (both cold).
        // Token 1 re-touches both, promoting them to t2 (p stays 0,
        // t2 = [X, Y] — Case I hits on both). Token 2 is the hazard: A is
        // a total miss (Case IV) that evicts an unprotected t2 resident
        // and lands in t1; B, in the SAME token as A, is also a total
        // miss — ARC's ordinary per-request algorithm alone would pick
        // A (t1's sole, and therefore LRU, resident) as B's eviction
        // victim, which production physically cannot survive: a token's
        // forward pass encodes every routed expert's GEMV into ONE Metal
        // command buffer before committing (module doc comment;
        // moe_expert_cache.rs:47-58, 887-912).
        let capacity = 2;
        let mut arc = ArcPolicy::new(capacity);
        let (x, y, a, b) = (10usize, 20usize, 30usize, 40usize);
        use AccessOutcome::{Hit, Miss};

        arc.begin_token();
        assert_eq!(arc.access(x), Miss);
        assert_eq!(arc.access(y), Miss);
        assert_eq!(dq(&arc.t1), vec![x, y]);

        arc.begin_token();
        assert_eq!(arc.access(x), Hit);
        assert_eq!(arc.access(y), Hit);
        assert!(
            arc.t1.is_empty(),
            "both promoted out of t1 by their 2nd touch"
        );
        assert_eq!(
            dq(&arc.t2),
            vec![x, y],
            "both promoted to t2 (frequency-protected)"
        );

        arc.begin_token();
        assert_eq!(arc.access(a), Miss);
        assert_eq!(arc.access(b), Miss);
        assert_eq!(
            dq(&arc.t1),
            vec![a, b],
            "A must survive B's SAME-token admission — production cannot evict a slot \
             resolved earlier this token"
        );
        assert!(arc.t2.is_empty());
        assert_eq!(arc.evictions(), 2, "X and Y evicted from t2, not A");

        // Physically-achievable next-token outcome: both A and B, having
        // survived their shared token, are ordinary cache hits.
        arc.begin_token();
        assert_eq!(arc.access(a), Hit);
        assert_eq!(arc.access(b), Hit);
    }

    // -----------------------------------------------------------------
    // Finding 5: hand-computed ARC B1/B2 ghost-hit adaptation trace,
    // asserting p, T1/T2/B1/B2 contents, and victim choice at each step —
    // not just the resident-capacity invariant `arc_never_exceeds_capacity`
    // checks (which a sign-flipped or victim-swapped adaptation could
    // still satisfy).
    // -----------------------------------------------------------------

    #[test]
    fn arc_ghost_hit_adaptation_hand_computed() {
        // capacity=2. One access per token (begin_token before each) —
        // orthogonal to the same-token guard above; isolates ARC's
        // ordinary Case I-IV adaptation logic.
        let capacity = 2;
        let mut arc = ArcPolicy::new(capacity);
        use AccessOutcome::{Hit, Miss};

        // t1=[1] (cold miss).
        arc.begin_token();
        assert_eq!(arc.access(1), Miss);
        assert_eq!(dq(&arc.t1), vec![1]);

        // t1=[1,2] (cold miss, cache now full).
        arc.begin_token();
        assert_eq!(arc.access(2), Miss);
        assert_eq!(dq(&arc.t1), vec![1, 2]);

        // Case I (t1 hit): 1 promoted to t2. t1=[2], t2=[1].
        arc.begin_token();
        assert_eq!(arc.access(1), Hit);
        assert_eq!(dq(&arc.t1), vec![2]);
        assert_eq!(dq(&arc.t2), vec![1]);

        // Case IV (t1_b1 < c, total == c): evicts 2 (t1's only resident)
        // into b1. t1=[3], t2=[1], b1=[2]. p unchanged by this branch.
        arc.begin_token();
        assert_eq!(arc.access(3), Miss);
        assert_eq!(dq(&arc.t1), vec![3]);
        assert_eq!(dq(&arc.t2), vec![1]);
        assert_eq!(dq(&arc.b1), vec![2]);
        assert_eq!(arc.evictions(), 1);
        assert_eq!(arc.p, 0);

        // Case I (t2 hit, re-promote 1): no structural change beyond Hit.
        arc.begin_token();
        assert_eq!(arc.access(1), Hit);

        // Case II: B1 ghost hit on 2. ratio = (|b2|=0 / |b1|=1).max(1) =
        // 1. p = (0 + 1).min(2) = 1 — B1 hit must ADAPT p UP (favor t1/
        // recency). evict_from_t1 = t1.len()(1) > p(1)? false; tie clause
        // needs triggered_by_b2_hit which is false here -> evict from t2
        // instead: victim = 1 (t2's only resident) -> b2=[1]. 2 leaves
        // b1 and is re-admitted into t2 (not t1).
        arc.begin_token();
        assert_eq!(arc.access(2), Miss);
        assert_eq!(arc.p, 1, "B1 ghost hit must increase p toward t1/recency");
        assert!(arc.b1.is_empty(), "2 must leave b1 once re-admitted");
        assert_eq!(
            dq(&arc.t2),
            vec![2],
            "2 re-admitted into t2, not t1 (ARC Case II)"
        );
        assert_eq!(dq(&arc.t1), vec![3], "t1 untouched by this Case II call");
        assert_eq!(
            dq(&arc.b2),
            vec![1],
            "evicted victim (1, from t2) becomes a b2 ghost"
        );
        assert_eq!(arc.evictions(), 2);

        // Case III: B2 ghost hit on 1. ratio = (|b1|=0 / |b2|=1).max(1) =
        // 1. p = 1.saturating_sub(1) = 0 — B2 hit must ADAPT p DOWN (favor
        // t2/frequency). evict_from_t1 = t1.len()(1) > p(0)? true ->
        // evict t1's only resident (3) into b1. 1 leaves b2 and is
        // re-admitted into t2.
        arc.begin_token();
        assert_eq!(arc.access(1), Miss);
        assert_eq!(arc.p, 0, "B2 ghost hit must decrease p toward t2/frequency");
        assert!(arc.b2.is_empty(), "1 must leave b2 once re-admitted");
        assert_eq!(dq(&arc.t2), vec![2, 1], "1 re-admitted into t2");
        assert!(
            arc.t1.is_empty(),
            "t1's only resident (3) was the Case III victim"
        );
        assert_eq!(
            dq(&arc.b1),
            vec![3],
            "evicted victim (3, from t1) becomes a b1 ghost"
        );
        assert_eq!(arc.evictions(), 3);
    }

    // -----------------------------------------------------------------
    // FreqAdmissionPolicy
    // -----------------------------------------------------------------

    #[test]
    fn freq_admission_admits_on_second_touch() {
        let mut policy = FreqAdmissionPolicy::new(4, 100);
        use AccessOutcome::{Hit, Miss};
        assert_eq!(policy.access(42), Miss); // 1st touch: never admitted
        assert_eq!(policy.access(42), Miss); // 2nd touch: admits, still a miss
        assert_eq!(policy.access(42), Hit); // 3rd touch: now resident
    }

    #[test]
    fn freq_admission_window_expiry_resets_eligibility() {
        let mut policy = FreqAdmissionPolicy::new(4, 2);
        use AccessOutcome::{Hit, Miss};
        assert_eq!(policy.access(1), Miss); // pos=1, A first touch
        assert_eq!(policy.access(2), Miss); // pos=2, filler
        assert_eq!(policy.access(3), Miss); // pos=3, filler
        // pos=4: gap since A's last touch (pos 1) is 3 > window(2) — NOT
        // admitted, treated as a fresh first touch.
        assert_eq!(policy.access(1), Miss);
        // pos=5: gap since pos 4 is 1 < window(2) — admitted now.
        assert_eq!(policy.access(1), Miss);
        // pos=6: resident from the previous access — hit.
        assert_eq!(policy.access(1), Hit);
    }

    // -----------------------------------------------------------------
    // Finding 2 regression: gap == window is OUTSIDE the sliding window
    // (off-by-one), not the boundary-inclusive edge.
    // -----------------------------------------------------------------

    #[test]
    fn freq_admission_gap_equal_to_window_is_not_eligible() {
        let mut policy = FreqAdmissionPolicy::new(4, 2);
        use AccessOutcome::{Hit, Miss};
        assert_eq!(policy.access(1), Miss); // pos=1: A's first touch.
        assert_eq!(policy.access(2), Miss); // pos=2: filler, unrelated id.
        // pos=3: gap since A's pos-1 touch is exactly 2 == window. A
        // 2-access window ending at pos 3 covers positions {2, 3};
        // position 1 is outside it, so A must NOT be admitted here. With
        // the old (buggy) `<=` comparator this WOULD admit A.
        assert_eq!(policy.access(1), Miss);
        // pos=4: if pos 3 had (incorrectly) admitted A, this would be a
        // Hit (A already resident). It must be another Miss, proving A
        // is still not resident after the gap==window touch.
        assert_eq!(policy.access(1), Miss);
        // pos=4's own gap (4-3=1 < window) DOES admit A — pos=5 confirms.
        assert_eq!(policy.access(1), Hit);
    }

    // -----------------------------------------------------------------
    // Finding 1 audit (per its closing instruction: "Do the same audit
    // for every future non-LRU policy"): FreqAdmissionPolicy needs no
    // explicit same-token guard — proof trace for the struct doc
    // comment's claim.
    // -----------------------------------------------------------------

    #[test]
    fn freq_admission_never_evicts_a_same_token_touched_id_without_an_explicit_guard() {
        let capacity = 2;
        let window = 100;
        let records = vec![
            record(0, 0, &[100]),      // P: first touch
            record(0, 1, &[200]),      // Q: first touch
            record(0, 2, &[100]),      // P: admitted
            record(0, 3, &[200]),      // Q: admitted -> resident = {P, Q}, cache full
            record(0, 4, &[300]),      // A: first touch
            record(0, 5, &[400]),      // B: first touch
            record(0, 6, &[300, 400]), // SAME token: A then B, both admitted here
        ];
        let mut policy = FreqAdmissionPolicy::new(capacity, window);
        for r in &records {
            policy.begin_token();
            for &id in &r.selected_ids {
                policy.access(id);
            }
        }
        assert!(
            policy.resident.contains(&300) && policy.resident.contains(&400),
            "A (300) and B (400), both touched in the SAME final token, must both \
             survive that token"
        );
        assert_eq!(policy.evictions(), 2, "P and Q evicted, not A or B");
    }

    #[test]
    fn freq_admission_evicts_lru_among_admitted() {
        let mut policy = FreqAdmissionPolicy::new(2, 100);
        // Admit 1 and 2 (each needs 2 touches).
        for id in [1, 2, 1, 2] {
            policy.access(id);
        }
        assert_eq!(policy.evictions(), 0);
        // Admit 3 (2 touches) — cache is full (1, 2 resident), must evict
        // the LRU of the two: 1 was touched at pos 3 (its 2nd touch),
        // 2 at pos 4 (its 2nd touch), so 1 is LRU.
        policy.access(3);
        policy.access(3);
        assert_eq!(policy.evictions(), 1);
        // 1 was evicted to make room for 3 — a fresh access is a miss.
        assert_eq!(policy.access(1), AccessOutcome::Miss);
    }

    // -----------------------------------------------------------------
    // Simulation driver / reporting
    // -----------------------------------------------------------------

    #[test]
    fn run_simulation_rejects_num_slots_below_top_k() {
        let layers = vec![(0usize, vec![record(0, 0, &[1, 2, 3])])];
        let cfg = SimConfig {
            num_slots: 2,
            top_k: 3,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap_err();
        assert!(err.contains("num_slots"), "unexpected error: {err}");
    }

    // -----------------------------------------------------------------
    // Finding 3 regression: zero-valued sizing inputs must fail closed as
    // an ordinary simulation error, not a policy-constructor panic.
    // -----------------------------------------------------------------

    #[test]
    fn run_simulation_rejects_zero_num_slots() {
        let layers = vec![(0usize, vec![record(0, 0, &[])])];
        let cfg = SimConfig {
            num_slots: 0,
            top_k: 0,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap_err();
        assert!(err.contains("num_slots"), "unexpected error: {err}");
    }

    #[test]
    fn run_simulation_rejects_zero_top_k() {
        let layers = vec![(0usize, vec![record(0, 0, &[])])];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 0,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap_err();
        assert!(err.contains("top_k"), "unexpected error: {err}");
    }

    #[test]
    fn run_simulation_rejects_zero_num_slots_and_top_k_together() {
        // A one-record trace with `selected_ids: []` and `--num-slots 0 --top-k 0`.
        let layers = vec![(0usize, vec![record(0, 0, &[])])];
        let cfg = SimConfig {
            num_slots: 0,
            top_k: 0,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap_err();
        assert!(err.contains("num_slots"), "unexpected error: {err}");
    }

    #[test]
    fn run_simulation_rejects_zero_freq_admission_window() {
        let layers = vec![(0usize, vec![record(0, 0, &[1])])];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 1,
        };
        let err = run_simulation(
            &layers,
            &cfg,
            &[PolicySpec::Lru, PolicySpec::FreqAdmission { window: 0 }],
        )
        .unwrap_err();
        assert!(err.contains("window"), "unexpected error: {err}");
    }

    // -----------------------------------------------------------------
    // Finding 4 regression: reported raw counts are per single LOGICAL
    // cache, not the two physical ExpertSlotCache instances (gate_up +
    // down) production runs per layer.
    // -----------------------------------------------------------------

    #[test]
    fn overall_stats_are_per_single_logical_cache_not_physical_two_cache_totals() {
        // Cold one-layer, one-token, one-expert trace: production would
        // perform this exact miss in BOTH the gate_up and down
        // ExpertSlotCache for this layer (identical selected_ids
        // sequence) — two physical misses. This simulator models one
        // logical cache per layer, so it reports ONE.
        let layers = vec![(0usize, vec![record(0, 0, &[42])])];
        let cfg = SimConfig {
            num_slots: 1,
            top_k: 1,
        };
        let reports = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap();
        assert_eq!(
            reports[0].overall.misses, 1,
            "logical single-cache count, not the physical x2 production performs"
        );
        assert_eq!(reports[0].overall.hits, 0);
        assert_eq!(
            reports[0].overall.evictions, 0,
            "first-ever access into an empty slot pool is never an eviction"
        );

        // Forced eviction: capacity=1, two distinct ids across two
        // tokens — one logical eviction here; production performs this
        // same eviction independently in both its gate_up and down
        // caches (physical x2, logical x1).
        let layers2 = vec![(0usize, vec![record(0, 0, &[1]), record(0, 1, &[2])])];
        let reports2 = run_simulation(&layers2, &cfg, &[PolicySpec::Lru]).unwrap();
        assert_eq!(reports2[0].overall.evictions, 1);
    }

    #[test]
    fn run_simulation_rejects_missing_baseline() {
        let layers = vec![(0usize, vec![record(0, 0, &[1])])];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 1,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Arc]).unwrap_err();
        assert!(err.contains("baseline"), "unexpected error: {err}");
    }

    #[test]
    fn run_simulation_rejects_selected_ids_exceeding_top_k() {
        let layers = vec![(0usize, vec![record(0, 0, &[1, 2, 3])])];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 2,
        };
        let err = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap_err();
        assert!(err.contains("exceeds --top-k"), "unexpected error: {err}");
    }

    #[test]
    fn run_simulation_delta_vs_baseline_is_zero_for_baseline_itself() {
        let layers = vec![(0usize, vec![record(0, 0, &[1, 2])])];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 2,
        };
        let reports = run_simulation(&layers, &cfg, &[PolicySpec::Lru]).unwrap();
        assert_eq!(reports.len(), 1);
        assert!((reports[0].delta_vs_baseline_pp).abs() < 1e-9);
    }

    #[test]
    fn format_table_includes_every_policy_and_layer() {
        let layers = vec![
            (0usize, vec![record(0, 0, &[1, 2])]),
            (1usize, vec![record(1, 0, &[3])]),
        ];
        let cfg = SimConfig {
            num_slots: 4,
            top_k: 2,
        };
        let reports = run_simulation(&layers, &cfg, &[PolicySpec::Lru, PolicySpec::Arc]).unwrap();
        let table = format_table(&reports);
        assert!(table.contains("lru (baseline)"));
        assert!(table.contains("arc"));
        assert!(table.contains("layer"));
    }
}
