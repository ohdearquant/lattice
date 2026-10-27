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
    let records = read_trace(jsonl.as_bytes()).expect("valid trace parses (blank line skipped)");
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
