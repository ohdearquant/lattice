const EOS: u32 = 2;

fn is_stop(id: u32) -> bool {
    id == EOS
}

// -----------------------------------------------------------------------
// Plain-greedy oracle: check is_stop, then push, bounded by max_len.
// Stop-token contract (#613): the stop token is never in the output — it
// still consumes the position (an "attempt" within max_len).
// Independent reference: do NOT derive from self-spec logic.
// -----------------------------------------------------------------------
fn plain_greedy_oracle(logit_rows: &[Vec<f32>], max_len: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for row in logit_rows.iter().take(max_len) {
        let token = crate::speculative::argmax(row) as u32;
        if is_stop(token) {
            break;
        }
        out.push(token);
    }
    out
}

// Regression test (#806): the oracle's argmax must be first-wins, matching
// `crate::speculative::argmax`'s documented tie-break contract. A last-wins
// oracle would silently mask a first-wins regression in the production
// Metal decode path whenever two logits tie exactly.
#[test]
fn plain_greedy_oracle_argmax_is_first_wins_on_ties() {
    let logit_rows = vec![vec![1.0, 1.0, 0.0]];
    let out = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(
        out,
        vec![0],
        "tie must resolve to the first index, not the last"
    );
}

// -----------------------------------------------------------------------
// Self-spec simulation: encodes the fixed per-round decision logic of
// `generate_greedy_self_spec` as pure token operations, no GPU.
//
// Alignment with the real loop and the oracle:
//   logit_rows[i] = target's logit vector for position i.
//   argmax(logit_rows[0])   = prefill argmax (first pending).
//   argmax(logit_rows[i+1]) = target's prediction at pos i+1, used both as
//                             verify target and as next_pending on full accept.
//
// `draft_seq[i]` = the draft proposal at round i.
//
// Each round:
//   - pending_token = logit_rows[pos] argmax
//   - verify: target agrees if draft_seq[round] == argmax(logit_rows[pos+1])
//   - on accept: push pending + all agreeing drafts, next_pending = logit_rows[last+1]
//   - on reject: push pending, target's replacement = argmax(logit_rows[pos+1])
//   - fallback paths behave like the plain-greedy single-step decode
//
// Termination for all paths: stop-token contract (#613) — the terminating
// token is excluded from `out`, never pushed, regardless of budget.
// -----------------------------------------------------------------------
fn simulate_self_spec(
    logit_rows: &[Vec<f32>],
    // draft_seq[i]: what the draft proposes at round i
    draft_seq: &[u32],
    // simulate_fallback: if true, bypass the normal verify path and use the
    // fallback (single-step decode) path, so we can exercise cases FB.
    simulate_fallback: bool,
    max_len: usize,
) -> Vec<u32> {
    assert!(!logit_rows.is_empty(), "need at least one logit row");
    let mut out: Vec<u32> = Vec::new();
    let mut pos = 0usize; // logit_rows index for current pending

    // Case A: prefill argmax is stop. Stop-token contract (#613):
    // excluded from the output unconditionally — the real function's
    // max_new_tokens==0 guard already returns separately (Length)
    // before this check ever runs, so budget is never the reason this
    // branch produces an empty output.
    let pending_first = crate::speculative::argmax(&logit_rows[pos]) as u32;
    if is_stop(pending_first) {
        return out;
    }

    let mut pending_token = pending_first;
    let mut round = 0usize;

    'round: loop {
        if out.len() >= max_len {
            break;
        }

        if simulate_fallback || pos + 1 >= logit_rows.len() || round >= draft_seq.len() {
            // Fallback path (single-step decode): push pending, check next.
            // Mirrors no-checkpoint / checkpoint-failure / verify-failure sites.
            out.push(pending_token);
            if out.len() >= max_len || pos + 1 >= logit_rows.len() {
                break;
            }
            let next = crate::speculative::argmax(&logit_rows[pos + 1]) as u32;
            // Stop-token contract (#613): never push the terminating token.
            if is_stop(next) {
                break;
            }
            if out.len() >= max_len {
                break;
            }
            pending_token = next;
            pos += 1;
            round += 1;
            continue 'round;
        }

        let draft = draft_seq[round];
        round += 1;

        // Verify: target agrees with draft if argmax(logit_rows[pos+1]) == draft.
        let target_at_pos1 = crate::speculative::argmax(&logit_rows[pos + 1]) as u32;
        let accepted = target_at_pos1 == draft;

        if accepted {
            // pending_token is always pushed first.
            out.push(pending_token);
            // Stop-token contract (#613): never push draft when it is
            // itself the stop — independent of whether the cap is hit.
            if is_stop(draft) {
                break;
            }
            if out.len() >= max_len {
                break;
            }
            out.push(draft);
            if out.len() >= max_len {
                break;
            }
            // next_pending = target prediction after the last accepted token.
            let next_pending_pos = pos + 2;
            if next_pending_pos >= logit_rows.len() {
                break;
            }
            let next_pending = crate::speculative::argmax(&logit_rows[next_pending_pos]) as u32;
            // Stop-token contract (#613): never push next_pending when
            // it is itself the stop.
            if is_stop(next_pending) {
                break;
            }
            if out.len() >= max_len {
                break;
            }
            pending_token = next_pending;
            pos += 2;
        } else {
            // Rejection: push pending, target's replacement is argmax(logit_rows[pos+1]).
            out.push(pending_token);
            let replacement = target_at_pos1;
            // Stop-token contract (#613): never push the replacement
            // when it is itself the stop.
            if is_stop(replacement) {
                break;
            }
            if out.len() >= max_len {
                break;
            }
            pending_token = replacement;
            pos += 1;
        }
    }
    out
}

fn make_logit(winner: u32) -> Vec<f32> {
    let size = 100usize;
    let mut v = vec![0.0f32; size];
    if (winner as usize) < size {
        v[winner as usize] = 10.0;
    }
    v
}

// -----------------------------------------------------------------------
// Case A: prefill argmax is stop. Stop-token contract (#613): oracle and
// sim both emit [] — the terminating token is excluded, generated_tokens = 0.
// -----------------------------------------------------------------------
#[test]
fn test_case_a_prefill_eos() {
    let logit_rows = vec![make_logit(EOS), make_logit(10)];
    let draft_seq = vec![10u32];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(
        oracle,
        Vec::<u32>::new(),
        "oracle sanity: prefill EOS yields []"
    );

    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(
        sim,
        Vec::<u32>::new(),
        "case A: sim must emit [] (stop excluded, #613)"
    );
    assert_eq!(oracle, sim, "case A: oracle == sim required");
}

// -----------------------------------------------------------------------
// Case A with max_len = 0: no budget → emit nothing (mirrors MTP
// max_new_tokens=0 guard). Stop-token contract (#613): identical output
// to `test_case_a_prefill_eos` above ([] either way) — under the
// exclude contract, Case A no longer has budget-dependent *token*
// behavior; only `stopped`/`stop_reason` still distinguish zero-budget
// (Length) from stop-at-first-token (Eos) — see
// `test_self_spec_stopped_case_a_*` below for that distinction.
// -----------------------------------------------------------------------
#[test]
fn test_case_a_prefill_eos_zero_budget() {
    let logit_rows = vec![make_logit(EOS), make_logit(10)];
    let draft_seq = vec![10u32];
    let oracle = plain_greedy_oracle(&logit_rows, 0);
    assert_eq!(oracle, Vec::<u32>::new(), "oracle: zero budget → empty");
    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, 0);
    assert_eq!(sim, Vec::<u32>::new(), "case A zero budget: must be empty");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Rejection path: target replacement is stop. Stop-token contract
// (#613): oracle/sim both emit [pending] — the replacement is excluded.
// -----------------------------------------------------------------------
#[test]
fn test_rejection_eos() {
    // pos0→10 (pending), pos1→EOS (target; draft proposes 99 = wrong).
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let draft_seq = vec![99u32];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10], "rejection-EOS oracle: [10]");

    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(sim, vec![10], "rejection-EOS: sim must match oracle");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Rejection path: budget clips the stop token (max_len = 1).
// Oracle stops at 1 token; sim must also stop at 1 (do not exceed budget).
// -----------------------------------------------------------------------
#[test]
fn test_rejection_eos_budget_clips_stop() {
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let draft_seq = vec![99u32];
    let oracle = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(oracle, vec![10], "budget-1: oracle emits only pending");
    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, 1);
    assert_eq!(sim, vec![10], "budget-1: sim must not emit stop beyond cap");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Full-accept path: next_pending is stop. Stop-token contract (#613):
// oracle/sim both emit [10, 20] — next_pending is excluded.
// -----------------------------------------------------------------------
#[test]
fn test_full_accept_next_pending_eos() {
    // pos0→10 (pending), pos1→20 (draft=20 accepted), pos2→EOS (next_pending).
    let logit_rows = vec![make_logit(10), make_logit(20), make_logit(EOS)];
    let draft_seq = vec![20u32];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10, 20], "full-accept-EOS oracle: [10, 20]");

    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(sim, vec![10, 20], "full-accept-EOS: sim must match oracle");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Full-accept path: budget clips EOS (max_len = 2).
// -----------------------------------------------------------------------
#[test]
fn test_full_accept_next_pending_eos_budget_clips() {
    let logit_rows = vec![make_logit(10), make_logit(20), make_logit(EOS)];
    let draft_seq = vec![20u32];
    let oracle = plain_greedy_oracle(&logit_rows, 2);
    assert_eq!(oracle, vec![10, 20], "budget-2: oracle stops before EOS");
    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, 2);
    assert_eq!(sim, vec![10, 20], "budget-2: sim must not exceed cap");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Fallback path (case FB): next token from single-step decode is stop.
// Stop-token contract (#613): oracle/sim both emit [10] — next excluded.
// -----------------------------------------------------------------------
#[test]
fn test_fallback_path_eos() {
    // pos0→10 (pending), pos1→EOS (single-step decode result).
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let draft_seq = vec![]; // unused in fallback mode
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10], "fallback-EOS oracle: [10]");

    let sim = simulate_self_spec(&logit_rows, &draft_seq, true, max_len);
    assert_eq!(sim, vec![10], "fallback-EOS: sim must match oracle");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Fallback path: budget clips the stop token (max_len = 1).
// -----------------------------------------------------------------------
#[test]
fn test_fallback_path_eos_budget_clips() {
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let draft_seq = vec![];
    let oracle = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(oracle, vec![10]);
    let sim = simulate_self_spec(&logit_rows, &draft_seq, true, 1);
    assert_eq!(sim, vec![10], "fallback-EOS budget-1: must not exceed cap");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Accept+draft-is-stop: this site name predates #613 ("already correct"
// referred to the pre-#613 include-contract, where this specific site
// pushed the stop-carrying draft even before the #237 EOS-faithfulness
// fixes landed elsewhere). Stop-token contract (#613): oracle/sim now
// both emit [pending] — the accepted draft (itself a stop) is excluded.
// -----------------------------------------------------------------------
#[test]
fn test_accept_draft_is_stop_already_correct() {
    // pos0→10 (pending), pos1→EOS (draft=EOS accepted by target).
    let logit_rows = vec![make_logit(10), make_logit(EOS), make_logit(30)];
    let draft_seq = vec![EOS];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10], "accept+draft-stop oracle: [10]");

    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(sim, vec![10], "accept+draft-stop: draft excluded (#613)");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// No-stop normal sequence: all accepted, no EOS.  Oracle and sim agree.
// -----------------------------------------------------------------------
#[test]
fn test_all_accepted_no_stop() {
    let logit_rows = vec![
        make_logit(10),
        make_logit(20),
        make_logit(30),
        make_logit(40),
    ];
    let draft_seq = vec![20u32, 30, 40];
    let max_len = 4;
    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(
        oracle, sim,
        "all-accept non-stop: oracle={oracle:?} sim={sim:?}"
    );
}

// -----------------------------------------------------------------------
// Mixed: some accepted, some rejected, EOS at the end via rejection path.
// Stop-token contract (#613): oracle/sim agree fully, EOS excluded.
// -----------------------------------------------------------------------
#[test]
fn test_mixed_accept_reject_then_rejection_eos() {
    // target: 10 → 20 → 30 → EOS
    let logit_rows = vec![
        make_logit(10),
        make_logit(20),
        make_logit(30),
        make_logit(EOS),
    ];
    // draft: accept 20 at round0, reject 99 at round1 (target says 30).
    // Then target says EOS at pos2 → reject-EOS path.
    let draft_seq = vec![20u32, 99];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10, 20, 30]);

    let sim = simulate_self_spec(&logit_rows, &draft_seq, false, max_len);
    assert_eq!(sim, vec![10, 20, 30], "mixed reject-EOS: got {sim:?}");
    assert_eq!(oracle, sim);
}

// -----------------------------------------------------------------------
// Seeded randomised sweep: oracle vs sim must agree across all EOS positions.
// Covers all termination paths probabilistically.
// -----------------------------------------------------------------------
#[test]
fn test_randomised_sweep() {
    let mut rng = Lcg::new(0xfeedface_deadbeef);
    let vocab_size = 8usize; // EOS=2
    let trials = 400;

    for trial in 0..trials {
        let seq_len = (rng.next_u64() % 6 + 2) as usize;
        let max_len = 20;
        let fallback = rng.next_u64().is_multiple_of(4); // 25% fallback

        let logit_rows: Vec<Vec<f32>> = (0..=seq_len)
            .map(|_| {
                let winner = (rng.next_u64() % vocab_size as u64) as usize;
                let mut row = vec![0.0f32; vocab_size];
                row[winner] = 10.0;
                row
            })
            .collect();
        let draft_seq: Vec<u32> = (0..seq_len)
            .map(|_| (rng.next_u64() % vocab_size as u64) as u32)
            .collect();

        let oracle = plain_greedy_oracle(&logit_rows, max_len);
        let sim = simulate_self_spec(&logit_rows, &draft_seq, fallback, max_len);

        assert_eq!(
            oracle,
            sim,
            "trial {trial} (fallback={fallback}): oracle != sim\n  winners={:?}\n  draft={:?}\n  oracle={:?}\n  sim={:?}",
            logit_rows
                .iter()
                .map(|r| crate::speculative::argmax(r) as u32)
                .collect::<Vec<_>>(),
            draft_seq,
            oracle,
            sim,
        );
    }
}

// -----------------------------------------------------------------------
// Seeded randomised sweep with small caps: stresses the budget-clip paths.
// -----------------------------------------------------------------------
#[test]
fn test_randomised_sweep_small_cap() {
    let mut rng = Lcg::new(0xabcd_1234_5678_ef01);
    let vocab_size = 6usize; // EOS=2, frequent stops
    let trials = 400;

    for trial in 0..trials {
        let seq_len = (rng.next_u64() % 6 + 2) as usize;
        let max_len = (rng.next_u64() % 4 + 1) as usize; // 1..=4
        let fallback = rng.next_u64().is_multiple_of(4);

        let logit_rows: Vec<Vec<f32>> = (0..=seq_len)
            .map(|_| {
                let winner = (rng.next_u64() % vocab_size as u64) as usize;
                let mut row = vec![0.0f32; vocab_size];
                row[winner] = 10.0;
                row
            })
            .collect();
        let draft_seq: Vec<u32> = (0..seq_len)
            .map(|_| (rng.next_u64() % vocab_size as u64) as u32)
            .collect();

        let oracle = plain_greedy_oracle(&logit_rows, max_len);
        let sim = simulate_self_spec(&logit_rows, &draft_seq, fallback, max_len);

        assert!(
            sim.len() <= max_len,
            "trial {trial}: sim exceeded cap {max_len}: {sim:?}"
        );
        assert_eq!(
            oracle,
            sim,
            "trial {trial} (max_len={max_len}, fallback={fallback}): oracle != sim\n  winners={:?}\n  draft={:?}\n  oracle={:?}\n  sim={:?}",
            logit_rows
                .iter()
                .map(|r| crate::speculative::argmax(r) as u32)
                .collect::<Vec<_>>(),
            draft_seq,
            oracle,
            sim,
        );
    }
}

/// Minimal deterministic LCG — shared with `mtp_greedy_round_tests`.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

// -----------------------------------------------------------------------
// FIX 2 & FIX 3: `stopped` semantics for fallback and accepted-draft paths.
//
// FIX 2 (five fallback/rejection/full-accept sites in generate_greedy_self_spec):
//   Pre-fix: `stopped = true` was placed OUTSIDE the budget guard, so a clipped
//   stop (budget already full) incorrectly set stopped=true. Fix: move
//   `stopped = true` INSIDE the `generated_ids.len() < max_new_tokens` guard.
//
// FIX 3 (accepted-draft stop, inside `for (i, &draft) in draft_tokens...`):
//   Pre-fix: `break` only exited the inner draft loop, leaving the outer 'round
//   loop to continue an extra iteration without recording stopped=true. Fix:
//   `stopped = true; break 'round;` to exit both loops and record the stop.
//
// Stop-token contract (#613): every terminating token below (`next`, `draft`
// when it's the stop, `next_pending`, `replacement`) is now EXCLUDED from
// `out` — never pushed. This does not change the FIX-2/FIX-3 `stopped`
// gating logic itself (still `stopped = true` inside the same budget
// guard); it only removes the push that used to accompany it. Concretely
// this means `tokens` in every test below drops its trailing EOS but the
// `stopped` boolean is UNCHANGED at all 4 existing boundaries (verified by
// trace: the guard `out.len() < max_len` is evaluated at the same point,
// over the same `out.len()`, in both the pre-#613 and post-#613 code —
// only the push after the guard is removed).
//
// `generate_greedy_self_spec` requires Metal hardware (GPU). These tests drive
// the identical decision logic via simulate_self_spec_with_stopped.
//
// Mutation-sensitivity for FIX 2: move `stopped = true` back outside the guard
// in simulate_self_spec_with_stopped → test_self_spec_stopped_fallback_eos_clipped
// goes RED (expected false, got true).
//
// Mutation-sensitivity for FIX 3: replace `break 'round` with plain `break` and
// remove `stopped = true` from simulate_self_spec_with_stopped →
// test_self_spec_stopped_accepted_draft_stop goes RED (expected true, got false).
// -----------------------------------------------------------------------
fn simulate_self_spec_with_stopped(
    logit_rows: &[Vec<f32>],
    draft_seq: &[u32],
    simulate_fallback: bool,
    max_len: usize,
) -> (Vec<u32>, bool) {
    assert!(!logit_rows.is_empty(), "need at least one logit row");
    let mut out: Vec<u32> = Vec::new();
    let mut stopped = false;
    let mut pos = 0usize;

    let pending_first = crate::speculative::argmax(&logit_rows[pos]) as u32;
    if is_stop(pending_first) {
        // Stop-token contract (#613): excluded from `out` regardless of
        // budget. `stopped` still reflects whether budget was available
        // (mirrors max_new_tokens==0 short-circuiting to Length upstream
        // of this check in the real function).
        if max_len > 0 {
            stopped = true;
        }
        return (out, stopped);
    }

    let mut pending_token = pending_first;
    let mut round = 0usize;

    'round: loop {
        if out.len() >= max_len {
            break;
        }

        if simulate_fallback || pos + 1 >= logit_rows.len() || round >= draft_seq.len() {
            out.push(pending_token);
            // Mirror production check order: is_stop(next) is evaluated BEFORE the
            // budget guard, exactly as in generate_greedy_self_spec. This makes the
            // simulation mutation-sensitive for FIX 2: when budget is exactly full
            // after pushing pending, the stop token is clipped → stopped stays false.
            if pos + 1 < logit_rows.len() {
                let next = crate::speculative::argmax(&logit_rows[pos + 1]) as u32;
                if is_stop(next) {
                    // Stop-token contract (#613): never push `next`.
                    // FIX 2: stopped=true only when budget allowed
                    // reaching this decision (not clipped).
                    if out.len() < max_len {
                        stopped = true;
                    }
                    break;
                }
                // Not a stop: continue if budget allows.
                if out.len() < max_len {
                    pending_token = next;
                    pos += 1;
                    round += 1;
                    continue 'round;
                }
            }
            break;
        }

        let draft = draft_seq[round];
        round += 1;

        let target_at_pos1 = crate::speculative::argmax(&logit_rows[pos + 1]) as u32;
        let accepted = target_at_pos1 == draft;

        if accepted {
            out.push(pending_token);
            if is_stop(draft) {
                // Stop-token contract (#613): never push draft. FIX 3:
                // still record stop and exit both loops when budget
                // allowed reaching this decision.
                if out.len() < max_len {
                    stopped = true;
                }
                break 'round;
            }
            if out.len() >= max_len {
                break;
            }
            out.push(draft);
            if out.len() >= max_len {
                break;
            }
            let next_pending_pos = pos + 2;
            if next_pending_pos >= logit_rows.len() {
                break;
            }
            let next_pending = crate::speculative::argmax(&logit_rows[next_pending_pos]) as u32;
            if is_stop(next_pending) {
                // Stop-token contract (#613): never push next_pending.
                // FIX 2 (full-accept path): stopped only when budget
                // allowed reaching this decision.
                if out.len() < max_len {
                    stopped = true;
                }
                break;
            }
            if out.len() >= max_len {
                break;
            }
            pending_token = next_pending;
            pos += 2;
        } else {
            out.push(pending_token);
            let replacement = target_at_pos1;
            if is_stop(replacement) {
                // Stop-token contract (#613): never push the
                // replacement. FIX 2 (rejection path): stopped only
                // when budget allowed reaching this decision.
                if out.len() < max_len {
                    stopped = true;
                }
                break;
            }
            if out.len() >= max_len {
                break;
            }
            pending_token = replacement;
            pos += 1;
        }
    }
    (out, stopped)
}

// FIX 2: fallback path, budget not full → stop reached (not clipped) → stopped=true.
#[test]
fn test_self_spec_stopped_fallback_eos_emitted() {
    // Fallback: pending=10, next=EOS (excluded, #613). Budget=10 → stopped=true.
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &[], true, 10);
    assert_eq!(tokens, vec![10]);
    assert!(stopped, "stop reached (not clipped) → stopped must be true");
}

// FIX 2: fallback path, budget already full after pushing pending → stop clipped → stopped=false.
#[test]
fn test_self_spec_stopped_fallback_eos_clipped() {
    // Fallback: pending=10, next=EOS. Budget=1. After pushing pending, out.len()=1 >= max_len=1.
    // The stop is never reached (clipped) → stopped=false.
    // Pre-fix (stopped=true outside guard): this test goes RED.
    let logit_rows = vec![make_logit(10), make_logit(EOS)];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &[], true, 1);
    assert_eq!(tokens, vec![10]);
    assert!(
        !stopped,
        "stop clipped by budget cap in fallback path → stopped must be false (length cap)"
    );
}

// FIX 3: accepted draft is stop token, budget allows → stopped=true (draft excluded, #613).
#[test]
fn test_self_spec_stopped_accepted_draft_stop() {
    // logit_rows[0]=10 (pending), logit_rows[1]=EOS (target agrees with draft=EOS).
    // draft accepted, is_stop(draft)=true, budget=10 → stopped=true (draft excluded).
    // Pre-fix (plain `break` without `stopped=true`): this test goes RED.
    let logit_rows = vec![make_logit(10), make_logit(EOS), make_logit(30)];
    let draft_seq = vec![EOS];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &draft_seq, false, 10);
    assert_eq!(tokens, vec![10]);
    assert!(
        stopped,
        "accepted draft is stop token → stopped must be true"
    );
}

// FIX 3: accepted draft is stop, but budget is already full after pending → draft clipped → stopped=false.
#[test]
fn test_self_spec_stopped_accepted_draft_stop_clipped() {
    // pending=10 pushed (budget=1 full), draft=EOS never reached → stopped=false.
    let logit_rows = vec![make_logit(10), make_logit(EOS), make_logit(30)];
    let draft_seq = vec![EOS];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &draft_seq, false, 1);
    assert_eq!(tokens, vec![10]);
    assert!(
        !stopped,
        "accepted draft stop clipped by budget cap → stopped must be false"
    );
}

// Case A (prefill argmax is stop), budget available → stopped=true (#613:
// excluded from `out`, but the round genuinely stopped, not capped).
// Closes a pre-existing gap: no test previously exercised Case A through
// `simulate_self_spec_with_stopped` (only the token-level oracle covered
// it). Mutation-sensitivity: remove `stopped = true` from the `is_stop
// (pending_first)` branch → this test goes RED (expected true, got false).
#[test]
fn test_self_spec_stopped_case_a_prefill_eos() {
    let logit_rows = vec![make_logit(EOS), make_logit(10)];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &[10], false, 10);
    assert_eq!(tokens, Vec::<u32>::new());
    assert!(
        stopped,
        "prefill stop with budget available → stopped must be true"
    );
}

// Case A with zero budget: mirrors the real function's separate
// max_new_tokens==0 early return (Length, not Eos) — stopped=false.
#[test]
fn test_self_spec_stopped_case_a_zero_budget() {
    let logit_rows = vec![make_logit(EOS), make_logit(10)];
    let (tokens, stopped) = simulate_self_spec_with_stopped(&logit_rows, &[10], false, 0);
    assert_eq!(tokens, Vec::<u32>::new());
    assert!(
        !stopped,
        "zero budget → stopped must be false (Length, not Eos)"
    );
}
