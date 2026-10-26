use super::{MtpRoundOutcome, mtp_greedy_round};

const EOS: u32 = 2;

fn is_stop(id: u32) -> bool {
    id == EOS
}

// -----------------------------------------------------------------------
// Plain-greedy oracle: given a sequence of per-position logit vectors,
// greedily pick argmax at each position, stop when argmax is a stop
// token (excluding that stop token from the output), or at max_len.
//
// Stop-token contract (#613): the
// is_stop check happens BEFORE the push, so a stop token is never
// present in `token_ids`/`text`. The stop still consumes the position
// (an "attempt" within max_len), it's just not emitted.
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
// MTP simulation: drives `mtp_greedy_round` with the same synthetic
// target logits used by the oracle.
//
// Alignment with the oracle:
//   logit_rows[i] = target's logit vector for position i.
//   argmax(logit_rows[0]) = first pending token (prefill argmax).
//   argmax(logit_rows[i+1]) = target's prediction for pos+1 from pos i.
//
// Each MTP round at pos i:
//   pending          = argmax(logit_rows[i])
//   draft            = draft_seq[i]
//   target_logits_0  = logit_rows[i+1]   (verify pos i+1)
//   accepted         = draft == argmax(target_logits_0)
//   bonus            = accepted ? argmax(logit_rows[i+2])
//                               : argmax(logit_rows[i+1])
//
// Case A (pending is stop): stop without emitting — stop-token contract
// (#613): the terminating token is excluded from the output. See the
// contract documented on `GenerateOutput`.
// -----------------------------------------------------------------------
fn simulate_mtp(logit_rows: &[Vec<f32>], draft_seq: &[u32], max_len: usize) -> Vec<u32> {
    assert!(
        !logit_rows.is_empty(),
        "simulate_mtp: need at least one logit row"
    );
    let mut out: Vec<u32> = Vec::new();
    let mut pos = 0usize; // index into logit_rows for the current pending
    let mut draft_idx = 0usize; // sequential index into draft_seq

    loop {
        if out.len() >= max_len {
            break;
        }
        let pending_token = crate::speculative::argmax(&logit_rows[pos]) as u32;

        // Case A: pending is a stop token. Stop-token contract (#613):
        // excluded from the output. For subsequent rounds,
        // mtp_greedy_round guarantees EmitAndContinue only returns
        // non-stop next_pending, so this branch fires only at pos=0.
        if is_stop(pending_token) {
            break;
        }

        // Need at least logit_rows[pos+1] for verify.
        if pos + 1 >= logit_rows.len() {
            out.push(pending_token);
            break;
        }

        let draft_token = if draft_idx < draft_seq.len() {
            draft_seq[draft_idx]
        } else {
            // No more drafts: fall back to pushing pending and stopping.
            out.push(pending_token);
            break;
        };
        draft_idx += 1;

        let target_logits_0 = &logit_rows[pos + 1];
        let accepted = draft_token == crate::speculative::argmax(target_logits_0) as u32;
        let bonus_token = if accepted {
            if pos + 2 < logit_rows.len() {
                crate::speculative::argmax(&logit_rows[pos + 2]) as u32
            } else {
                0 // no further prediction available; use 0 (non-stop)
            }
        } else {
            crate::speculative::argmax(target_logits_0) as u32
        };

        match mtp_greedy_round(pending_token, draft_token, accepted, bonus_token, is_stop) {
            MtpRoundOutcome::EmitAndStop(tokens) => {
                // Mirror the real call site: clip the emission to the remaining
                // budget so MTP never exceeds max_len (plain greedy caps exactly).
                // `tokens` already excludes the stop token itself (#613).
                let remaining = max_len - out.len();
                out.extend_from_slice(&tokens[..tokens.len().min(remaining)]);
                break;
            }
            MtpRoundOutcome::EmitAndContinue { emit, next_pending } => {
                let remaining = max_len - out.len();
                out.extend_from_slice(&emit[..emit.len().min(remaining)]);
                if out.len() >= max_len {
                    break;
                }
                // Advance pos: accept advances by 2, reject by 1.
                pos = if accepted { pos + 2 } else { pos + 1 };
                let _ = next_pending; // value verified via logit_rows[pos]
            }
        }
    }
    out
}

// -----------------------------------------------------------------------
// Case A: prefill argmax is a stop token.
// Stop-token contract (#613): oracle and MTP both emit [] — the
// terminating token is excluded, generated_tokens = 0.
// -----------------------------------------------------------------------
#[test]
fn test_case_a_prefill_eos() {
    // logit_rows[0] argmax is EOS — the very first pending is a stop.
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(EOS, 5), // pos0 → EOS
        make_logit(10, 5),  // pos1 → 10 (never reached)
    ];
    let draft_seq = vec![10u32]; // draft never consulted
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    // Oracle: EOS excluded → [].
    assert_eq!(
        oracle,
        Vec::<u32>::new(),
        "oracle sanity: prefill EOS yields []"
    );

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    // Fixed MTP: pending_first = EOS → emit [], return.
    assert_eq!(
        mtp,
        Vec::<u32>::new(),
        "case A: mtp must emit [] (stop excluded, #613)"
    );
    assert_eq!(
        oracle, mtp,
        "case A: full oracle==mtp equality required; mtp={mtp:?}"
    );
}

// -----------------------------------------------------------------------
// Case B: accept + bonus is stop.
// Stop-token contract (#613): oracle and MTP both emit [pending, draft]
// — the bonus stop token is excluded.
// -----------------------------------------------------------------------
#[test]
fn test_case_b_accept_bonus_stop() {
    // pending=10, draft=20 (accepted), bonus=EOS.
    // accept: target[1]=20, target[2]=EOS.
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10 (pending)
        make_logit(20, 5),  // pos1 → 20 (target agrees with draft)
        make_logit(EOS, 5), // pos2 → EOS (bonus)
    ];
    let draft_seq = vec![20u32];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    // Oracle: push 10, push 20, EOS excluded, stop → [10, 20].
    assert_eq!(oracle, vec![10, 20], "case B oracle sanity");

    // Verify the pure function directly.
    let outcome = mtp_greedy_round(
        10,   // pending
        20,   // draft (accepted)
        true, // accepted
        EOS,  // bonus is stop
        is_stop,
    );
    assert_eq!(
        outcome,
        MtpRoundOutcome::EmitAndStop(vec![10, 20]),
        "case B: accept+bonus-stop must emit [pending, draft] (bonus excluded, #613)"
    );

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    assert_eq!(mtp, vec![10, 20], "case B: mtp must match oracle");
    assert_eq!(
        oracle, mtp,
        "case B: full oracle==mtp equality required; mtp={mtp:?}"
    );
}

// -----------------------------------------------------------------------
// Case C: reject + replacement is stop.
// Stop-token contract (#613): oracle and MTP both emit [pending] — the
// replacement stop token is excluded.
// -----------------------------------------------------------------------
#[test]
fn test_case_c_reject_bonus_stop() {
    // pending=10, draft=99 (wrong), target says EOS at pos1.
    // reject: target[1]=EOS, bonus=EOS.
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10 (pending)
        make_logit(EOS, 5), // pos1 → EOS (target's own choice = EOS)
    ];
    let draft_seq = vec![99u32]; // wrong draft
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    // Oracle: push 10, EOS excluded, stop → [10].
    assert_eq!(oracle, vec![10], "case C oracle sanity");

    // Verify the pure function directly.
    let outcome = mtp_greedy_round(
        10,    // pending
        99,    // draft (rejected)
        false, // not accepted
        EOS,   // bonus = target's replacement = EOS
        is_stop,
    );
    assert_eq!(
        outcome,
        MtpRoundOutcome::EmitAndStop(vec![10]),
        "case C: reject+bonus-stop must emit [pending] (bonus excluded, #613)"
    );

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    assert_eq!(mtp, vec![10], "case C: mtp must match oracle");
    assert_eq!(
        oracle, mtp,
        "case C: full oracle==mtp equality required; mtp={mtp:?}"
    );
}

// -----------------------------------------------------------------------
// Hand-constructed: All non-stop tokens, draft always accepted.
// Oracle and MTP should agree on the first `max_len` tokens.
// -----------------------------------------------------------------------
#[test]
fn test_all_accepted_no_stop() {
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5), // pos0 → 10
        make_logit(20, 5), // pos1 → 20
        make_logit(30, 5), // pos2 → 30
        make_logit(40, 5), // pos3 → 40
    ];
    let draft_seq = vec![20u32, 30, 40];
    let max_len = 4;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    assert_eq!(
        oracle, mtp,
        "all-accept non-stop: oracle={oracle:?} mtp={mtp:?}"
    );
}

// -----------------------------------------------------------------------
// Hand-constructed: Draft always rejected, no stop tokens.
// Oracle and MTP output must agree.
// -----------------------------------------------------------------------
#[test]
fn test_all_rejected_no_stop() {
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),
        make_logit(20, 5),
        make_logit(30, 5),
        make_logit(40, 5),
    ];
    let draft_seq = vec![99u32, 99, 99];
    let max_len = 4;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    assert_eq!(
        oracle, mtp,
        "all-reject non-stop: oracle={oracle:?} mtp={mtp:?}"
    );
}

// -----------------------------------------------------------------------
// Hand-constructed: target emits EOS via the accept+bonus-stop path.
// Stop-token contract (#613): oracle and MTP both emit [10, 20].
// -----------------------------------------------------------------------
#[test]
fn test_target_eos_via_bonus() {
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10
        make_logit(20, 5),  // pos1 → 20 (draft correct → accepted)
        make_logit(EOS, 5), // pos2 → EOS (bonus)
    ];
    let draft_seq = vec![20u32, 99];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10, 20], "oracle: [10, 20]");

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    // Round 0: pending=10, draft=20, accepted, bonus=EOS
    //   → mtp_greedy_round(10, 20, true, EOS, is_stop)
    //   → EmitAndStop([10, 20])   (case B, bonus excluded per #613)
    assert_eq!(mtp, vec![10, 20], "after case B fix, mtp must equal oracle");
    assert_eq!(oracle, mtp);
}

// -----------------------------------------------------------------------
// KEY REGRESSION TEST (concern #2 of #237):
// Draft proposes EOS but the target argmax is NOT a stop token.
// OLD CODE: short-circuit fires, emits [pending], stops early → WRONG.
// NEW CODE: verify runs, draft rejected, continue with target replacement.
// -----------------------------------------------------------------------
#[test]
fn test_draft_eos_but_target_non_stop() {
    // Sequence: target wants 10 → 20 → 30 → EOS.
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10
        make_logit(20, 5),  // pos1 → 20  (draft proposes EOS here — WRONG)
        make_logit(30, 5),  // pos2 → 30
        make_logit(EOS, 5), // pos3 → EOS
    ];
    let draft_seq = vec![EOS, 99u32, 99];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10, 20, 30], "oracle sanity check");

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    // Round 0: pending=10, draft=EOS, target[1]=20 (non-stop) → rejected.
    //   bonus = 20.  EmitAndContinue([10], 20).  pos→1.
    // Round 1: pending=20, draft=99, target[2]=30 → rejected.
    //   bonus = 30.  EmitAndContinue([20], 30).  pos→2.
    // Round 2: pending=30, draft=99, target[3]=EOS → rejected.
    //   bonus = EOS.  EmitAndStop([30])   ← case C fix, EOS excluded (#613).
    // MTP output = [10, 20, 30].
    assert_eq!(
        mtp,
        vec![10, 20, 30],
        "draft-EOS-but-target-non-stop: must match oracle; got {mtp:?}"
    );
    assert_eq!(oracle, mtp);
    // Confirm it does NOT equal [10] (old short-circuit regression).
    assert_ne!(
        mtp,
        vec![10],
        "output must not be [10] (old short-circuit regression)"
    );
}

// -----------------------------------------------------------------------
// Hand-constructed: Draft proposes EOS and target AGREES.
// Stop-token contract (#613): MTP must emit [pending] and stop
// (accept+draft-is-stop path, draft/stop excluded).
// -----------------------------------------------------------------------
#[test]
fn test_draft_eos_accepted() {
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10 (pending)
        make_logit(EOS, 5), // pos1 → EOS (target agrees with draft)
        make_logit(30, 5),  // pos2 → 30 (bonus, irrelevant once EOS accepted)
    ];
    let draft_seq = vec![EOS];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10]);

    let outcome = mtp_greedy_round(
        10,   // pending
        EOS,  // draft (accepted: target[1]=EOS)
        true, // accepted
        30,   // bonus (irrelevant — draft is the stop)
        is_stop,
    );
    assert_eq!(
        outcome,
        MtpRoundOutcome::EmitAndStop(vec![10]),
        "accepted draft-EOS must emit [pending] and stop (draft excluded, #613)"
    );

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    assert_eq!(mtp, vec![10], "accepted draft-EOS: mtp={mtp:?}");
    assert_eq!(oracle, mtp);
}

// -----------------------------------------------------------------------
// Mixed: some accepted, some rejected, EOS at the end via case C.
// After fixes, oracle and MTP agree fully, EOS excluded (#613).
// -----------------------------------------------------------------------
#[test]
fn test_mixed_accept_reject_then_eos() {
    let logit_rows: Vec<Vec<f32>> = vec![
        make_logit(10, 5),  // pos0 → 10
        make_logit(20, 5),  // pos1 → 20 (draft correct)
        make_logit(30, 5),  // pos2 → 30 (draft wrong)
        make_logit(40, 5),  // pos3 → 40 (draft wrong)
        make_logit(EOS, 5), // pos4 → EOS
    ];
    let draft_seq = vec![20u32, 99, 99];
    let max_len = 10;

    let oracle = plain_greedy_oracle(&logit_rows, max_len);
    assert_eq!(oracle, vec![10, 20, 30, 40]);

    let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);
    // Round 0: pending=10, draft=20, accepted, bonus=30 (non-stop)
    //   → EmitAndContinue([10, 20], 30), pos=2
    // Round 1: pending=30, draft=99, rejected, bonus=40 (non-stop)
    //   → EmitAndContinue([30], 40), pos=3
    // Round 2: pending=40, draft=99, rejected, bonus=EOS (stop)
    //   → EmitAndStop([40])   ← case C fix, EOS excluded (#613)
    // MTP output = [10, 20, 30, 40].
    assert_eq!(
        mtp,
        vec![10, 20, 30, 40],
        "mixed: mtp must fully match oracle"
    );
    assert_eq!(oracle, mtp);
}

// -----------------------------------------------------------------------
// CAP-EDGE REGRESSION (PR #287):
// A round can emit up to 2 non-stop tokens (case B, #613). When
// max_new_tokens lands mid-emission, the call site must clip so MTP
// never exceeds the cap — plain greedy `generate()` stops at the cap
// even before an EOS.
// -----------------------------------------------------------------------
#[test]
fn test_cap_clips_case_b_before_bonus() {
    // Case B logits: [10, 20, EOS]; uncapped MTP would emit [10, 20]
    // (bonus EOS excluded either way, #613).
    let logit_rows: Vec<Vec<f32>> = vec![make_logit(10, 5), make_logit(20, 5), make_logit(EOS, 5)];
    let draft_seq = vec![20u32];

    // max_len = 2: cap lands exactly at the non-stop emission length.
    let oracle = plain_greedy_oracle(&logit_rows, 2);
    assert_eq!(oracle, vec![10, 20], "oracle caps at 2 tokens, no EOS");
    let mtp = simulate_mtp(&logit_rows, &draft_seq, 2);
    assert_eq!(
        mtp,
        vec![10, 20],
        "case B must clip to [10, 20] at max_len=2"
    );
    assert_eq!(oracle, mtp);

    // max_len = 1: only the pending token fits.
    let oracle1 = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(oracle1, vec![10]);
    let mtp1 = simulate_mtp(&logit_rows, &draft_seq, 1);
    assert_eq!(mtp1, vec![10], "case B must clip to [10] at max_len=1");
    assert_eq!(oracle1, mtp1);
}

#[test]
fn test_cap_clips_case_c_max_len_1() {
    // Case C: pending=10, draft wrong, target replacement is EOS.
    let logit_rows: Vec<Vec<f32>> = vec![make_logit(10, 5), make_logit(EOS, 5)];
    let draft_seq = vec![99u32];
    let oracle = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(oracle, vec![10]);
    let mtp = simulate_mtp(&logit_rows, &draft_seq, 1);
    assert_eq!(mtp, vec![10], "case C must clip to [10] at max_len=1");
    assert_eq!(oracle, mtp);
}

#[test]
fn test_cap_clips_continue_emit() {
    // accept + non-stop bonus → EmitAndContinue([pending, draft]); cap=1 clips to [pending].
    let logit_rows: Vec<Vec<f32>> = vec![make_logit(10, 5), make_logit(20, 5), make_logit(30, 5)];
    let draft_seq = vec![20u32, 30];
    let oracle = plain_greedy_oracle(&logit_rows, 1);
    assert_eq!(oracle, vec![10]);
    let mtp = simulate_mtp(&logit_rows, &draft_seq, 1);
    assert_eq!(
        mtp,
        vec![10],
        "EmitAndContinue must clip to [10] at max_len=1"
    );
    assert_eq!(oracle, mtp);
}

// -----------------------------------------------------------------------
// Randomised / property-style sweep: seeded RNG, arbitrary draft choices.
//
// With all three EOS-faithful fixes (A/B/C) and the stop-token exclusion
// contract (#613), `simulate_mtp` should produce token-for-token
// identical output to `plain_greedy_oracle` across all trials, including
// those where stop tokens appear at any position.
//
// The sweep also includes trials where logit_rows[0] is EOS (case A).
// max_len is set large (20) so that natural EOS-stop is reached before
// the cap, keeping EOS-faithfulness under test rather than cap behavior.
// -----------------------------------------------------------------------
#[test]
fn test_randomised_sweep() {
    let mut rng = Lcg::new(0xdeadbeef_cafebabe);

    let vocab_size = 8usize; // tiny vocab; EOS=2
    let trials = 400;

    for trial in 0..trials {
        let seq_len = (rng.next_u64() % 6 + 2) as usize; // 2..=7
        // Large max_len so EOS-stop is reached before cap in most trials.
        let max_len = 20;

        // Random logit rows — stop tokens may appear at any position.
        let logit_rows: Vec<Vec<f32>> = (0..=seq_len)
            .map(|_| {
                let winner = (rng.next_u64() % vocab_size as u64) as usize;
                let mut row = vec![0.0f32; vocab_size];
                row[winner] = 10.0;
                row
            })
            .collect();

        // Random draft sequence (may propose EOS at any position).
        let draft_seq: Vec<u32> = (0..seq_len)
            .map(|_| (rng.next_u64() % vocab_size as u64) as u32)
            .collect();

        let oracle = plain_greedy_oracle(&logit_rows, max_len);
        let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);

        assert_eq!(
            oracle,
            mtp,
            "trial {trial}: oracle != mtp\n  logit_winners={:?}\n  draft_seq={:?}\n  oracle={:?}\n  mtp={:?}",
            logit_rows
                .iter()
                .map(|r| crate::speculative::argmax(r) as u32)
                .collect::<Vec<_>>(),
            draft_seq,
            oracle,
            mtp
        );
    }
}

// -----------------------------------------------------------------------
// Randomised sweep with SMALL caps: stresses the emission-clip path so MTP
// never exceeds max_len even when a round wants to emit 2 tokens (#613:
// a round's non-stop payload is at most 2 tokens now, not 3).
// -----------------------------------------------------------------------
#[test]
fn test_randomised_sweep_small_cap() {
    let mut rng = Lcg::new(0x0123_4567_89ab_cdef);
    let vocab_size = 6usize; // EOS=2, frequent stops
    let trials = 400;

    for trial in 0..trials {
        let seq_len = (rng.next_u64() % 6 + 2) as usize; // 2..=7
        let max_len = (rng.next_u64() % 4 + 1) as usize; // 1..=4 — cap often bites mid-round

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
        let mtp = simulate_mtp(&logit_rows, &draft_seq, max_len);

        assert!(
            mtp.len() <= max_len,
            "trial {trial}: mtp exceeded cap {max_len}: {mtp:?}"
        );
        assert_eq!(
            oracle,
            mtp,
            "trial {trial} (max_len={max_len}): oracle != mtp\n  winners={:?}\n  draft={:?}\n  oracle={:?}\n  mtp={:?}",
            logit_rows
                .iter()
                .map(|r| crate::speculative::argmax(r) as u32)
                .collect::<Vec<_>>(),
            draft_seq,
            oracle,
            mtp
        );
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a logit vector of size 100 with `winner` at logit 10.0.
fn make_logit(winner: u32, _high: u32) -> Vec<f32> {
    let size = 100usize;
    let mut v = vec![0.0f32; size];
    if (winner as usize) < size {
        v[winner as usize] = 10.0;
    }
    v
}

/// Minimal deterministic LCG — no external crate needed.
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
// FIX 1: EmitAndStop `stopped` semantics.
//
// `generate_greedy_mtp` (under #[cfg(feature="metal-gpu")]) computes:
//
//   let emit_len = tokens.len().min(remaining);
//   generated_ids.extend_from_slice(&tokens[..emit_len]);
//   stopped = emit_len == tokens.len() && remaining > tokens.len();   // ← FIX 1
//
// The pre-fix code was `stopped = true` unconditionally, which is wrong when
// the budget clips the emission so not every committed token in `tokens`
// is actually pushed — in that case the reason for termination is the
// length cap, not the stop condition. Stop-token contract (#613): `tokens`
// never contains the stop token itself (only the committed non-stop
// tokens for the round), so this formula is unchanged by #613 — only the
// *meaning* of "every token in tokens was emitted" shifts, since `tokens`
// is shorter now (see the budget=2 test below for the concrete effect).
//
// #632: `emit_len == tokens.len()` alone under-counts one
// more boundary — when the committed payload exactly *fills* `remaining`
// (`remaining == tokens.len()`), the excluded stop token itself would sit
// one position beyond `max_new_tokens`. Plain greedy never reaches that
// position either (its loop caps at `max_new_tokens` decode steps), so it
// reports `Length`, not `Eos`, in that exact-fill case. The additional
// `remaining > tokens.len()` conjunct requires strictly more room than the
// committed payload — i.e. a slot left over for the excluded stop token —
// before calling it a genuine stop.
//
// `generate_greedy_mtp` requires Metal hardware and cannot be called from
// `cargo test` (no GPU). `simulate_mtp_with_stopped` mirrors the call-site
// logic exactly; tests here are mutation-sensitive against that logic.
//
// Mutation-sensitivity: change `stopped = emit_len == tokens.len() &&
// remaining > tokens.len()` in `simulate_mtp_with_stopped` below back to
// `stopped = emit_len == tokens.len()` (dropping the new conjunct) → the
// budget=2 exact-fill test goes RED (expected false, got true). Change it
// to `stopped = true` unconditionally → the budget=1 test also goes RED
// (expected false, got true).
// -----------------------------------------------------------------------
fn simulate_mtp_with_stopped(
    logit_rows: &[Vec<f32>],
    draft_seq: &[u32],
    max_len: usize,
) -> (Vec<u32>, bool) {
    let mut out: Vec<u32> = Vec::new();
    let mut stopped = false;
    let mut pos = 0usize;
    let mut draft_idx = 0usize;

    loop {
        if out.len() >= max_len {
            break;
        }
        let pending_token = crate::speculative::argmax(&logit_rows[pos]) as u32;
        if is_stop(pending_token) {
            // Stop-token contract (#613): excluded from `out`. The
            // loop-top guard already confirmed out.len() < max_len, so
            // budget was available for this token — a genuine stop.
            stopped = true;
            break;
        }
        if pos + 1 >= logit_rows.len() {
            out.push(pending_token);
            break;
        }
        let draft_token = if draft_idx < draft_seq.len() {
            draft_seq[draft_idx]
        } else {
            out.push(pending_token);
            break;
        };
        draft_idx += 1;

        let target_logits_0 = &logit_rows[pos + 1];
        let accepted = draft_token == crate::speculative::argmax(target_logits_0) as u32;
        let bonus_token = if accepted {
            if pos + 2 < logit_rows.len() {
                crate::speculative::argmax(&logit_rows[pos + 2]) as u32
            } else {
                0
            }
        } else {
            crate::speculative::argmax(target_logits_0) as u32
        };

        match mtp_greedy_round(pending_token, draft_token, accepted, bonus_token, is_stop) {
            MtpRoundOutcome::EmitAndStop(tokens) => {
                let remaining = max_len - out.len();
                let emit_len = tokens.len().min(remaining);
                out.extend_from_slice(&tokens[..emit_len]);
                // FIX 1: stopped only when every committed token was actually
                // emitted (not clipped) — tokens never includes the stop itself
                // — AND at least one more slot remained beyond the committed
                // payload for that excluded stop token (#632):
                // an exact-fill payload (remaining == tokens.len()) means the
                // stop token itself sits beyond the budget, so plain greedy
                // would report Length there, not Eos.
                stopped = emit_len == tokens.len() && remaining > tokens.len();
                break;
            }
            MtpRoundOutcome::EmitAndContinue { emit, next_pending } => {
                let remaining = max_len - out.len();
                out.extend_from_slice(&emit[..emit.len().min(remaining)]);
                if out.len() >= max_len {
                    break;
                }
                pos = if accepted { pos + 2 } else { pos + 1 };
                let _ = next_pending;
            }
        }
    }
    (out, stopped)
}

#[test]
fn test_mtp_stopped_stop_token_emitted() {
    // Budget=10: [10, 20] fit with room to spare (bonus EOS excluded
    // from `tokens`, #613) → emit_len=2 == tokens.len()=2 → stopped=true.
    let logit_rows = vec![make_logit(10, 5), make_logit(20, 5), make_logit(EOS, 5)];
    let draft_seq = vec![20u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 10);
    assert_eq!(tokens, vec![10, 20]);
    assert!(stopped, "stop reached (not clipped) → stopped must be true");
}

#[test]
fn test_mtp_stopped_stop_token_clipped_at_budget_2() {
    // Budget=2: EmitAndStop([10, 20]) — bonus EOS already excluded from
    // `tokens` (#613) — so emit_len=2 == tokens.len()=2, but `remaining`
    // (2) is NOT strictly greater than `tokens.len()` (2): the committed
    // payload exactly fills the budget, so the excluded bonus stop token
    // would have landed one position beyond `max_new_tokens`.
    //
    // #632: the prior expectation here
    // (`stopped=true`) was wrong: plain greedy's decode loop caps at
    // exactly `max_new_tokens` steps and never samples that extra
    // position, so it would report `Length`, not `Eos`, in this exact
    // scenario. `stopped=false` is the correct value — the emitted
    // tokens are still [10, 20] either way, only the boolean changes.
    let logit_rows = vec![make_logit(10, 5), make_logit(20, 5), make_logit(EOS, 5)];
    let draft_seq = vec![20u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 2);
    assert_eq!(tokens, vec![10, 20]);
    assert!(
        !stopped,
        "#632: tokens=[pending,draft] (bonus excluded) exactly fills budget=2 \
         with no slot left for the excluded stop token → stopped=false, matching \
         plain greedy's Length outcome at the same boundary"
    );
}

#[test]
fn test_mtp_stopped_accepted_draft_stop_budget_1() {
    // Accept case: the accepted draft token is itself the stop (not the
    // bonus). pending=10 (not stop), draft=EOS is accepted (target's
    // argmax at pos+1 agrees) → EmitAndStop([pending]) = [10] (draft
    // excluded, #613). Budget=1: remaining=1 == tokens.len()=1 (exact
    // fill) → the excluded stop token would sit one position beyond the
    // budget → stopped=false (#632 boundary case).
    let logit_rows = vec![make_logit(10, 5), make_logit(EOS, 5)];
    let draft_seq = vec![EOS];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 1);
    assert_eq!(tokens, vec![10]);
    assert!(
        !stopped,
        "accepted-draft-stop exactly fills budget=1 → stopped=false, not Eos"
    );
}

#[test]
fn test_mtp_stopped_accepted_draft_stop_budget_2() {
    // Same accept-draft-is-stop shape as above, but budget=2 leaves one
    // spare slot beyond the committed [pending] payload for the excluded
    // stop token → genuine stop, stopped=true.
    let logit_rows = vec![make_logit(10, 5), make_logit(EOS, 5)];
    let draft_seq = vec![EOS];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 2);
    assert_eq!(tokens, vec![10]);
    assert!(
        stopped,
        "accepted-draft-stop with budget to spare → stopped=true (genuine Eos)"
    );
}

#[test]
fn test_mtp_stopped_rejected_replacement_stop_budget_1() {
    // Reject case: draft is rejected (target's own argmax at pos+1
    // disagrees with the proposed draft), and the target's replacement
    // (bonus_token) is itself the stop → EmitAndStop([pending]) = [10]
    // (replacement excluded, #613). Budget=1: remaining=1 ==
    // tokens.len()=1 (exact fill) → stopped=false (#632).
    let logit_rows = vec![make_logit(10, 5), make_logit(EOS, 5)];
    // draft_seq deliberately does not match argmax(logit_rows[1])=EOS,
    // forcing rejection; the target's replacement (EOS) becomes bonus_token.
    let draft_seq = vec![99u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 1);
    assert_eq!(tokens, vec![10]);
    assert!(
        !stopped,
        "rejected-replacement-stop exactly fills budget=1 → stopped=false, not Eos"
    );
}

#[test]
fn test_mtp_stopped_rejected_replacement_stop_budget_2() {
    // Same rejected-replacement-is-stop shape as above, but budget=2
    // leaves a spare slot for the excluded stop token → genuine stop.
    let logit_rows = vec![make_logit(10, 5), make_logit(EOS, 5)];
    let draft_seq = vec![99u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 2);
    assert_eq!(tokens, vec![10]);
    assert!(
        stopped,
        "rejected-replacement-stop with budget to spare → stopped=true (genuine Eos)"
    );
}

#[test]
fn test_mtp_stopped_stop_token_clipped_at_budget_1() {
    // Budget=1: only the pending token fits; the round's bonus stop is
    // never reached → stopped=false. Unchanged by #613 (tokens=[10,20]
    // now instead of [10,20,EOS], but budget=1 clips before either).
    let logit_rows = vec![make_logit(10, 5), make_logit(20, 5), make_logit(EOS, 5)];
    let draft_seq = vec![20u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 1);
    assert_eq!(tokens, vec![10]);
    assert!(
        !stopped,
        "stop not reached within budget → stopped must be false"
    );
}

// Case A (prefill argmax is stop), budget available → stopped=true (#613:
// excluded from `out`, but the round genuinely stopped, not capped).
// Closes a pre-existing gap: no test previously exercised Case A through
// `simulate_mtp_with_stopped` (only the token-level oracle covered it).
// Mutation-sensitivity: remove `stopped = true` from the Case A branch
// → this test goes RED (expected true, got false).
#[test]
fn test_mtp_stopped_case_a_prefill_eos() {
    let logit_rows = vec![make_logit(EOS, 5), make_logit(10, 5)];
    let draft_seq = vec![10u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 10);
    assert_eq!(tokens, Vec::<u32>::new());
    assert!(
        stopped,
        "prefill stop with budget available → stopped must be true"
    );
}

// Case A with zero budget: the loop-top `out.len() >= max_len` guard
// fires before the Case A check is ever reached (mirrors the real
// function's separate max_new_tokens==0 early return: Length, not Eos).
#[test]
fn test_mtp_stopped_case_a_zero_budget() {
    let logit_rows = vec![make_logit(EOS, 5), make_logit(10, 5)];
    let draft_seq = vec![10u32];
    let (tokens, stopped) = simulate_mtp_with_stopped(&logit_rows, &draft_seq, 0);
    assert_eq!(tokens, Vec::<u32>::new());
    assert!(
        !stopped,
        "zero budget → stopped must be false (Length, not Eos)"
    );
}
