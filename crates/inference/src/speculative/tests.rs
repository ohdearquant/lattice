use super::*;

// -- NgramSpeculator unit tests --

#[test]
fn speculate_basic_ngram_match() {
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 6, 7], 5, 3);
    // Recent [2, 3, 4] matches prompt[1..4], should draft [5, 6, 7]
    let draft = spec.speculate(&[2, 3, 4]);
    assert_eq!(draft, vec![5, 6, 7]);
}

#[test]
fn speculate_no_match_returns_empty() {
    let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
    let draft = spec.speculate(&[10, 11, 12]);
    assert!(draft.is_empty());
}

#[test]
fn speculate_empty_recent_returns_empty() {
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
    let draft = spec.speculate(&[]);
    assert!(draft.is_empty());
}

#[test]
fn speculate_single_recent_token_skipped() {
    // Minimum n-gram is 2, so a single recent token cannot match.
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
    let draft = spec.speculate(&[2]);
    assert!(draft.is_empty());
}

#[test]
fn speculate_prefers_longest_match() {
    // Prompt: [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]
    // Trigram [3,4,5] matches at pos 2 (draft [3,4,5]->continuation)
    // and at pos 5 (draft [6,7]).
    // But if recent is [2,3,4,5], that's a 4-gram matching pos 1,
    // which should draft from pos 5 onward: [3, 4, 5].
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 3, 4, 5, 6, 7], 5, 3);
    // 4-gram [2,3,4,5] matches at pos 1, drafts prompt[5..8] = [3,4,5]
    let draft = spec.speculate(&[2, 3, 4, 5]);
    assert_eq!(draft, vec![3, 4, 5]);
}

#[test]
fn speculate_max_draft_limits_output() {
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 2);
    // Match [1, 2] at pos 0, max_draft=2 so drafts [3, 4] only
    let draft = spec.speculate(&[1, 2]);
    assert_eq!(draft, vec![3, 4]);
}

#[test]
fn speculate_saturates_adversarial_max_draft() {
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, usize::MAX);
    assert_eq!(spec.speculate(&[1, 2]), vec![3, 4]);
}

/// Fixture for the two clamp tests below.
///
/// Returns `(prompt, c_short, c_long)`, where `c_short` is the draft a
/// CLAMPED search produces and `c_long` the one an UNCLAMPED search
/// produces. The two tests are only able to fail because those differ, so
/// this helper asserts its own invariants rather than leaving them implicit
/// at two call sites: an edit that breaks the layout fails here, instead of
/// quietly leaving tests that pass while detecting nothing.
///
/// The clamp has to change a RESULT, or nothing distinguishes it from its
/// own absence. A prompt whose longest repeat is shorter than the limit
/// produces the same draft either way, so the layout is built the other way
/// round: a 70-token block appears twice, and its trailing 64 tokens also
/// appear on their own, earlier, with a different continuation.
///
/// Layout (indices), where `b` is 64 distinct tokens and `h` is a 6-token
/// head, so `hb` is 70 tokens long:
///
/// ```text
///   0..64    b            <- first occurrence of the 64-token suffix
///  64..68    c_short      <- what follows it
///  68..138   hb           <- first occurrence of the 70-token suffix
/// 138..142   c_long       <- what follows that
/// 142..212   hb           <- the tail, so the recent suffix IS hb
/// ```
///
/// Searching at n = 70 finds `hb` at 68 and drafts `c_long`. Clamped to 64
/// the search starts at the shorter suffix, finds `b` at 0 — earlier — and
/// drafts `c_short`.
fn clamp_fixture() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let b: Vec<u32> = (100..100 + MAX_NGRAM_LIMIT as u32).collect();
    let h: Vec<u32> = (200..206).collect();
    let c_short: Vec<u32> = vec![900, 901, 902, 903];
    let c_long: Vec<u32> = vec![910, 911, 912, 913];

    let hb: Vec<u32> = h.iter().chain(b.iter()).copied().collect();
    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend_from_slice(&b);
    prompt.extend_from_slice(&c_short);
    prompt.extend_from_slice(&hb);
    prompt.extend_from_slice(&c_long);
    prompt.extend_from_slice(&hb);

    assert_eq!(
        b.len(),
        MAX_NGRAM_LIMIT,
        "the short block must be exactly the limit, so a clamped search matches it whole"
    );
    assert!(
        hb.len() > MAX_NGRAM_LIMIT,
        "the long block must EXCEED the limit ({} vs {}), or clamped and unclamped searches \
             try the same lengths and neither test can fail",
        hb.len(),
        MAX_NGRAM_LIMIT
    );
    assert_eq!(
        &prompt[prompt.len() - hb.len()..],
        hb.as_slice(),
        "the prompt must end with the long block, or the recent suffix is not the one under test"
    );
    assert_ne!(
        c_short, c_long,
        "the two continuations must differ, or a clamped and an unclamped search draft the \
             same tokens and the assertions below hold with the clamp removed"
    );
    assert_eq!(
        c_short.len(),
        c_long.len(),
        "both continuations are used as the max_draft, so they must be the same length"
    );

    // The assertions above constrain the pieces. These constrain their PLACEMENT,
    // which is what the tests actually rest on: a layout edit that swaps or moves
    // a piece can leave every piece-level assertion and every `speculate_` test
    // passing while the fixture no longer exercises a distinct shorter match.
    let c_short_at = b.len();
    let hb_first_at = c_short_at + c_short.len();
    let c_long_at = hb_first_at + hb.len();
    let hb_last_at = c_long_at + c_long.len();

    assert_eq!(
        prompt.len(),
        hb_last_at + hb.len(),
        "the prompt must be exactly b | c_short | hb | c_long | hb; another length means a \
             piece was added, dropped or resized, and every offset below is then reading \
             something other than what it names"
    );
    assert_eq!(
        &prompt[..b.len()],
        b.as_slice(),
        "the prompt must OPEN with the bare {MAX_NGRAM_LIMIT}-token block. Opening with the \
             long block instead still contains a copy of it, so a clamped search still matches \
             and still drafts c_short: the test keeps passing while no longer exercising a \
             distinct shorter match"
    );
    assert_eq!(
        &prompt[c_short_at..c_short_at + c_short.len()],
        c_short.as_slice(),
        "the short block must be followed by c_short, or a clamped search drafts something \
             the clamp assertion does not name"
    );
    assert_eq!(
        &prompt[hb_first_at..hb_first_at + hb.len()],
        hb.as_slice(),
        "the long block must sit between the two continuations, or the unclamped search has \
             no earlier long match to find"
    );
    assert_eq!(
        &prompt[c_long_at..c_long_at + c_long.len()],
        c_long.as_slice(),
        "the long block must be followed by c_long. With c_short here instead, a clamped and \
             an unclamped search draft the SAME tokens, so removing the clamp stops changing the \
             outcome and the regression test can no longer fail"
    );

    (prompt, c_short, c_long)
}

/// Removing the clamp makes this fail: the unclamped search matches the
/// 70-token block and drafts the other continuation.
#[test]
fn speculate_clamps_max_ngram_to_the_limit() {
    let (prompt, c_short, c_long) = clamp_fixture();

    // Far above the limit: the stored value must come back clamped.
    let spec = NgramSpeculator::new(prompt.clone(), 10_000, c_short.len());
    let draft = spec.speculate(&prompt);

    assert_eq!(
        draft, c_short,
        "an unclamped search matches the longer block and drafts {c_long:?}; the clamp must \
             hold the search to {MAX_NGRAM_LIMIT} tokens, which matches earlier and drafts \
             {c_short:?}"
    );
}

/// A caller asking for exactly the limit and a caller asking for far more
/// must be indistinguishable, which is the property the clamp exists to
/// provide.
#[test]
fn speculate_at_the_limit_and_far_above_it_agree() {
    let (prompt, c_short, _c_long) = clamp_fixture();

    let at_limit =
        NgramSpeculator::new(prompt.clone(), MAX_NGRAM_LIMIT, c_short.len()).speculate(&prompt);
    let above = NgramSpeculator::new(prompt.clone(), usize::MAX, c_short.len()).speculate(&prompt);

    assert_eq!(at_limit, above);
    assert_eq!(at_limit, c_short);
}

#[test]
fn speculate_match_at_end_of_prompt_returns_empty() {
    // Match is at the very end of prompt -- nothing left to draft
    let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
    let draft = spec.speculate(&[2, 3]);
    assert!(draft.is_empty());
}

#[test]
fn speculate_match_near_end_returns_partial() {
    // Match [2, 3] at pos 1, only token 4 left to draft
    let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
    let draft = spec.speculate(&[2, 3]);
    assert_eq!(draft, vec![4]);
}

#[test]
fn speculate_max_ngram_clamped_to_recent_len() {
    // max_ngram=10 but only 3 recent tokens -- should still work
    let spec = NgramSpeculator::new(vec![5, 6, 7, 8, 9], 10, 3);
    let draft = spec.speculate(&[5, 6, 7]);
    assert_eq!(draft, vec![8, 9]);
}

// -- find_ngram tests --

#[test]
fn find_ngram_empty_pattern() {
    let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
    assert_eq!(spec.find_ngram(&[]), None);
}

#[test]
fn find_ngram_pattern_longer_than_prompt() {
    let spec = NgramSpeculator::new(vec![1, 2], 5, 3);
    assert_eq!(spec.find_ngram(&[1, 2, 3]), None);
}

#[test]
fn find_ngram_exact_match() {
    let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
    assert_eq!(spec.find_ngram(&[1, 2, 3]), Some(0));
}

#[test]
fn find_ngram_returns_first_occurrence() {
    let spec = NgramSpeculator::new(vec![1, 2, 1, 2, 5], 5, 3);
    assert_eq!(spec.find_ngram(&[1, 2]), Some(0));
}

// -- argmax tests --

#[test]
fn argmax_basic() {
    assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
}

#[test]
fn argmax_empty() {
    assert_eq!(argmax(&[]), 0);
}

#[test]
fn argmax_tie_first_wins() {
    // Greedy contract: speculation must be token-identical to plain greedy.
    // The engine greedy decode (metal_qwen35), the sampling module, and
    // torch.argmax all break ties toward the FIRST max index, so this helper
    // must too (see #280).
    assert_eq!(argmax(&[1.0, 1.0, 0.5]), 0);
}

#[test]
fn argmax_tie_matches_engine_first_wins() {
    // Cross-check against the engine's first-wins loop on a tie-heavy vector:
    // enabling speculation must not change the token on tied top logits.
    let tie = [0.0_f32, 0.0, -1.0, 0.0];
    let engine_first_wins = {
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in tie.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i;
            }
        }
        best_id
    };
    assert_eq!(engine_first_wins, 0);
    assert_eq!(argmax(&tie), engine_first_wins);
}

#[test]
fn argmax_skips_nan_and_never_panics() {
    // NaN loses to finite values; among finite ties the first index wins.
    // Matches sampling::argmax_f32_scalar and the engine greedy decode.
    assert_eq!(argmax(&[f32::NAN, 1.0, 1.0]), 1);
    assert_eq!(argmax(&[1.0, f32::NAN, 1.0]), 0);
    // All-NaN and all-(-inf) fall back to index 0 — must never panic.
    assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
}

#[test]
fn argmax_negative_values() {
    assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
}

// -- verify_draft tests --

#[test]
fn verify_draft_all_accepted() {
    // current=0, draft=[1,2,3].
    // Call 0: fwd(0, 10) → predicts draft[0]=1  → accepted=1, input=1
    // Call 1: fwd(1, 11) → predicts draft[1]=2  → accepted=2, input=2
    // Call 2: fwd(2, 12) → predicts draft[2]=3  → accepted=3, input=3
    // Call 3: fwd(3, 13) → accepted=3 == len=3  → break (bonus logits)
    // Result: accepted=3, logits.len()=4 (3 verification calls + 1 bonus call)
    let draft = vec![1u32, 2, 3];
    let mut call = 0usize;
    let (accepted, logits) = verify_draft(0, &draft, 10, |_tok, _pos| {
        let mut l = vec![0.0f32; 10];
        match call {
            0 => l[1] = 1.0, // predicts draft[0]=1
            1 => l[2] = 1.0, // predicts draft[1]=2
            2 => l[3] = 1.0, // predicts draft[2]=3
            _ => l[0] = 1.0, // bonus call — prediction is the next greedy token
        }
        call += 1;
        l
    });
    assert_eq!(accepted, 3);
    assert_eq!(logits.len(), 4);
}

#[test]
fn verify_draft_first_rejected() {
    // current=5, draft=[1,2,3].
    // Call 0: fwd(5, 0) → predicts 99 != draft[0]=1 → accepted=0, break
    // Result: accepted=0, logits.len()=1 (the rejection-correction call)
    let draft = vec![1u32, 2, 3];
    let (accepted, logits) = verify_draft(5, &draft, 0, |_tok, _pos| {
        let mut l = vec![0.0f32; 100];
        l[99] = 1.0;
        l
    });
    assert_eq!(accepted, 0);
    assert_eq!(logits.len(), 1);
    // The correction token is 99, not draft[0]=1
    assert_eq!(argmax(&logits[0]), 99);
}

#[test]
fn verify_draft_partial_acceptance() {
    // current=0, draft=[10,20,30,40].
    // Call 0: fwd(0,  0) → predicts 10  → accepted=1, input=10
    // Call 1: fwd(10, 1) → predicts 20  → accepted=2, input=20
    // Call 2: fwd(20, 2) → predicts 30  → accepted=3, input=30
    // Call 3: fwd(30, 3) → predicts  0 != draft[3]=40 → break
    // Result: accepted=3, logits.len()=4
    let draft = vec![10u32, 20, 30, 40];
    let mut call = 0usize;
    let (accepted, logits) = verify_draft(0, &draft, 0, |_tok, _pos| {
        let mut l = vec![0.0f32; 50];
        match call {
            0 => l[10] = 1.0, // agrees with draft[0]
            1 => l[20] = 1.0, // agrees with draft[1]
            2 => l[30] = 1.0, // agrees with draft[2]
            _ => l[0] = 1.0,  // disagrees with draft[3]=40
        }
        call += 1;
        l
    });
    assert_eq!(accepted, 3);
    assert_eq!(logits.len(), 4);
}

#[test]
fn verify_draft_single_token() {
    // current=7, draft=[42].
    // Call 0: fwd(7, 5) → predicts 42 == draft[0] → accepted=1, input=42
    // Call 1: fwd(42, 6) → accepted=1 == len=1 → break (bonus)
    // Result: accepted=1, logits.len()=2
    let draft = vec![42u32];
    let mut call = 0usize;
    let (accepted, logits) = verify_draft(7, &draft, 5, |_tok, _pos| {
        let mut l = vec![0.0f32; 100];
        if call == 0 {
            l[42] = 1.0; // predicts draft[0]=42
        } else {
            l[0] = 1.0; // bonus call
        }
        call += 1;
        l
    });
    assert_eq!(accepted, 1);
    assert_eq!(logits.len(), 2);
}

#[test]
fn verify_draft_empty() {
    // current=3, draft=[].
    // Call 0: fwd(3, 0) → accepted=0 == len=0 → break immediately (bonus logits)
    // Result: accepted=0, logits.len()=1
    let (accepted, logits) = verify_draft(3, &[], 0, |_tok, _pos| {
        let mut l = vec![0.0f32; 10];
        l[7] = 1.0;
        l
    });
    assert_eq!(accepted, 0);
    assert_eq!(logits.len(), 1);
    // The single logits entry is the greedy-correct next token distribution
    assert_eq!(argmax(&logits[0]), 7);
}

#[test]
fn verify_draft_positions_are_correct() {
    // current=9, draft=[1,2,3], position_start=100.
    // Call 0: fwd(9,   100) → predicts 1  → accepted=1
    // Call 1: fwd(1,   101) → predicts 2  → accepted=2
    // Call 2: fwd(2,   102) → predicts 3  → accepted=3
    // Call 3: fwd(3,   103) → bonus call  → break
    // Positions seen: [100, 101, 102, 103]
    let draft = vec![1u32, 2, 3];
    let mut positions = Vec::new();
    let mut call = 0usize;
    let _ = verify_draft(9, &draft, 100, |_tok, pos| {
        positions.push(pos);
        let mut l = vec![0.0f32; 10];
        match call {
            0 => l[1] = 1.0, // predicts draft[0]=1
            1 => l[2] = 1.0, // predicts draft[1]=2
            2 => l[3] = 1.0, // predicts draft[2]=3
            _ => l[0] = 1.0, // bonus
        }
        call += 1;
        l
    });
    assert_eq!(positions, vec![100, 101, 102, 103]);
}

// -- MTP verification tests --

/// Build logits where only token `tok` has a nonzero value.
fn logits_with_argmax(vocab: usize, tok: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; vocab];
    if (tok as usize) < vocab {
        v[tok as usize] = 1.0;
    }
    v
}

struct MockTargetVerifier {
    cache_pos: usize,
    logits_by_step: Vec<Vec<f32>>,
    calls: Vec<(Vec<u32>, usize)>,
}

impl MtpTargetVerifier for MockTargetVerifier {
    fn cache_position(&self) -> usize {
        self.cache_pos
    }
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        self.cache_pos = seq_len;
        Ok(())
    }
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        self.calls.push((tokens.to_vec(), start_pos));
        self.cache_pos += tokens.len();
        Ok(self.logits_by_step.clone())
    }
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        Vec::new()
    }
    fn restore_gdn_states(&mut self, _snapshot: &crate::attention::gdn::GdnSnapshot) {}
}

const EOS: u32 = 99;
const VOCAB: usize = 200;

#[test]
fn mtp_eos_mid_draft_accepts_through_eos_and_stops() {
    // Draft: [10, 11, EOS, 12]
    // Initial target argmax = 10 → first token accepted
    // target_logits[0] argmax = 11 → second accepted
    // target_logits[1] argmax = EOS → third accepted, EOS hit → stop
    let initial = logits_with_argmax(VOCAB, 10);
    let mut target = MockTargetVerifier {
        cache_pos: 5,
        logits_by_step: vec![
            logits_with_argmax(VOCAB, 11),
            logits_with_argmax(VOCAB, EOS),
            logits_with_argmax(VOCAB, 12),
        ],
        calls: vec![],
    };
    let draft = vec![10, 11, EOS, 12];

    let result =
        mtp_verify_precomputed_draft(draft, 5, &initial, Some(EOS), &mut target, 4).unwrap();

    assert_eq!(result.accepted_count, 3);
    assert_eq!(result.accepted_tokens, vec![10, 11, EOS]);
    assert!(result.stopped_by_eos);
    assert_eq!(result.fallback_token, None);
    assert_eq!(
        target.cache_pos,
        5 + 3,
        "cache must be rolled back to start + accepted"
    );
}

#[test]
fn mtp_position_accounting_rolls_back_after_partial_accept() {
    // Start cache at 100, draft 8, first 3 match, 4th mismatches
    let initial = logits_with_argmax(VOCAB, 1);
    let mut target = MockTargetVerifier {
        cache_pos: 100,
        logits_by_step: vec![
            logits_with_argmax(VOCAB, 2),  // agrees with draft[1]
            logits_with_argmax(VOCAB, 3),  // agrees with draft[2]
            logits_with_argmax(VOCAB, 77), // mismatches draft[3]=4
            logits_with_argmax(VOCAB, 5),
            logits_with_argmax(VOCAB, 6),
            logits_with_argmax(VOCAB, 7),
            logits_with_argmax(VOCAB, 8),
        ],
        calls: vec![],
    };
    let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    let result = mtp_verify_precomputed_draft(draft, 100, &initial, None, &mut target, 8).unwrap();

    assert_eq!(result.accepted_count, 3);
    // target cache rolled back to 100 + 3 = 103
    assert_eq!(target.cache_pos, 103, "target cache must be 100 + 3 = 103");
}

#[test]
fn mtp_partial_acceptance_returns_target_fourth_token() {
    // Draft: [1, 2, 3, 4, 5, 6, 7, 8]
    // initial argmax=1, target_logits[0] argmax=2, [1] argmax=3, [2] argmax=99 (mismatch with 4)
    let initial = logits_with_argmax(VOCAB, 1);
    let mut target = MockTargetVerifier {
        cache_pos: 0,
        logits_by_step: vec![
            logits_with_argmax(VOCAB, 2),
            logits_with_argmax(VOCAB, 3),
            logits_with_argmax(VOCAB, 99),
            logits_with_argmax(VOCAB, 5),
            logits_with_argmax(VOCAB, 6),
            logits_with_argmax(VOCAB, 7),
            logits_with_argmax(VOCAB, 8),
        ],
        calls: vec![],
    };
    let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    let result = mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 8).unwrap();

    assert_eq!(result.accepted_count, 3);
    assert_eq!(result.accepted_tokens, vec![1, 2, 3]);
    assert_eq!(result.fallback_token, Some(99));
}

#[test]
fn mtp_full_rejection_uses_initial_target_argmax() {
    // Draft: [42, 43], initial target argmax = 7 (mismatch with 42)
    let initial = logits_with_argmax(VOCAB, 7);
    let mut target = MockTargetVerifier {
        cache_pos: 0,
        logits_by_step: vec![],
        calls: vec![],
    };
    let draft = vec![42u32, 43];

    let result = mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 4).unwrap();

    assert_eq!(result.accepted_count, 0);
    assert_eq!(result.fallback_token, Some(7));
    assert!(
        target.calls.is_empty(),
        "verify_tokens must NOT be called on full rejection"
    );
}

#[test]
fn mtp_degenerate_draft_length_zero_or_one_uses_normal_decode() {
    let initial = logits_with_argmax(VOCAB, 5);
    for dl in [0usize, 1] {
        let mut target = MockTargetVerifier {
            cache_pos: 0,
            logits_by_step: vec![],
            calls: vec![],
        };
        let draft = vec![5u32]; // doesn't matter — degenerate path ignores draft
        let result =
            mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, dl).unwrap();
        assert_eq!(
            result.accepted_count, 0,
            "dl={dl}: degenerate path returns 0 accepted"
        );
        assert_eq!(
            result.fallback_token,
            Some(5),
            "dl={dl}: fallback must be initial target argmax"
        );
        assert!(
            target.calls.is_empty(),
            "dl={dl}: no verify_tokens calls in degenerate path"
        );
    }
}

#[test]
fn mtp_acceptance_metric_matches_manual_count() {
    // Draft length 8, accepted 6
    let initial = logits_with_argmax(VOCAB, 1);
    let mut target = MockTargetVerifier {
        cache_pos: 0,
        logits_by_step: vec![
            logits_with_argmax(VOCAB, 2),
            logits_with_argmax(VOCAB, 3),
            logits_with_argmax(VOCAB, 4),
            logits_with_argmax(VOCAB, 5),
            logits_with_argmax(VOCAB, 6),
            logits_with_argmax(VOCAB, 77), // mismatch with draft[6]=7
            logits_with_argmax(VOCAB, 8),
        ],
        calls: vec![],
    };
    let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

    let result = mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 8).unwrap();

    assert_eq!(result.accepted_count, 6);
    assert_eq!(result.metrics.draft_tokens, 8);
    assert_eq!(result.metrics.target_forwards, 1);
    assert_eq!(result.metrics.accepted_tokens, 6);
    assert!((result.metrics.acceptance_rate - 6.0 / 8.0).abs() < 1e-6);
    assert!((result.metrics.accepted_tokens_per_forward - 6.0).abs() < 1e-6);
}

#[test]
fn mtp_mock_forward_function_is_deterministic() {
    let initial = logits_with_argmax(VOCAB, 3);
    let make_target = || MockTargetVerifier {
        cache_pos: 10,
        logits_by_step: vec![
            logits_with_argmax(VOCAB, 4),
            logits_with_argmax(VOCAB, 5),
            logits_with_argmax(VOCAB, 99), // mismatch with draft[3]=6
        ],
        calls: vec![],
    };
    let draft1 = vec![3u32, 4, 5, 6];
    let draft2 = draft1.clone();

    let mut t1 = make_target();
    let mut t2 = make_target();
    let r1 = mtp_verify_precomputed_draft(draft1, 10, &initial, None, &mut t1, 4).unwrap();
    let r2 = mtp_verify_precomputed_draft(draft2, 10, &initial, None, &mut t2, 4).unwrap();

    assert_eq!(r1.accepted_count, r2.accepted_count);
    assert_eq!(r1.accepted_tokens, r2.accepted_tokens);
    assert_eq!(r1.fallback_token, r2.fallback_token);
}

// A verifier whose `verify_tokens` advances `cache_pos` then immediately
// returns an error — exercises the #282 error-path rollback in
// `mtp_verify_precomputed_draft` (and, by structural analogy, in
// `mtp_verify_draft`).  We use the precomputed helper because constructing a
// real `MtpVerifier` requires a live model.
struct ErroringTargetVerifier {
    cache_pos: usize,
}

impl MtpTargetVerifier for ErroringTargetVerifier {
    fn cache_position(&self) -> usize {
        self.cache_pos
    }
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        self.cache_pos = seq_len;
        Ok(())
    }
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        _start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        // Advance the cache before returning the error, mimicking a real
        // implementation that commits the forward pass then discovers a
        // shape mismatch in the output.
        self.cache_pos += tokens.len();
        Err(crate::error::InferenceError::Inference(
            "simulated verify_tokens failure".into(),
        ))
    }
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        Vec::new()
    }
    fn restore_gdn_states(&mut self, _snapshot: &crate::attention::gdn::GdnSnapshot) {}
}

#[test]
fn mtp_verify_error_restores_target_cache() {
    // Guards the #282 footgun: if `verify_tokens` advances the target cache
    // and then returns an error, the cache must be rolled back to its
    // pre-call position so subsequent calls see a consistent starting state.
    //
    // We use `mtp_verify_precomputed_draft` (the test helper) rather than
    // `mtp_verify_draft` because the latter requires a real `MtpVerifier`
    // backed by a model.  The rollback logic for the production path is
    // structurally identical.
    let target_start = 42usize;
    // initial argmax = draft[0] so execution reaches verify_tokens
    let initial = logits_with_argmax(VOCAB, 5);
    let draft = vec![5u32, 6, 7]; // draft[0]==5 matches initial argmax; not EOS

    let mut target = ErroringTargetVerifier {
        cache_pos: target_start,
    };

    let result = mtp_verify_precomputed_draft(draft, target_start, &initial, None, &mut target, 4);

    assert!(result.is_err(), "expected Err from erroring verifier");
    assert_eq!(
        target.cache_pos, target_start,
        "cache_pos must be restored to pre-call value on error (#282)"
    );
}

// -- generate_with_speculation integration tests --

#[test]
fn generate_no_speculation_possible() {
    // Prompt has no overlap with generated tokens -- falls back to normal decode.
    let prompt = vec![100u32, 101, 102];
    let eos = 999;
    let mut step = 0usize;
    let result = generate_with_speculation(
        &prompt,
        3,
        eos,
        |_tok, _pos| {
            step += 1;
            let mut l = vec![0.0f32; 200];
            // Generate tokens 200+step which won't match prompt
            let next = (50 + step).min(199);
            l[next] = 1.0;
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");
    assert_eq!(result.len(), 3);
}

#[test]
fn generate_stops_on_eos() {
    let prompt = vec![1u32, 2, 3];
    let eos = 99;
    let result = generate_with_speculation(
        &prompt,
        100,
        eos,
        |_tok, _pos| {
            let mut l = vec![0.0f32; 100];
            l[eos as usize] = 1.0; // always predict EOS
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");
    assert!(result.is_empty());
}

#[test]
fn generate_with_perfect_speculation() {
    // Prompt: [1, 2, 3, 4, 5, 6, 7]
    // After prefill, first call gets last token = 7, model predicts "1" so
    // all_tokens becomes [1,2,3,4,5,6,7,1].
    // Next iteration: recent tokens end with [7,1] -- matches prompt[6..8]?
    // No, prompt only has 7 tokens so [7,1] doesn't match.
    //
    // Better test: model copies from prompt.
    // Prompt: [10, 20, 30, 40, 50]
    // Model always predicts the token matching the prompt copy.
    let prompt = vec![10u32, 20, 30, 40, 50];
    let eos = 999;
    let mut call_count = 0usize;
    let result = generate_with_speculation(
        &prompt,
        5,
        eos,
        |_tok, _pos| {
            call_count += 1;
            let mut l = vec![0.0f32; 100];
            // Just produce non-EOS non-prompt tokens to avoid complicating logic
            l[60 + (call_count % 30)] = 1.0;
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");
    // Should produce exactly 5 tokens
    assert_eq!(result.len(), 5);
}

#[test]
fn generate_respects_max_new_tokens() {
    let prompt = vec![1u32, 2, 3];
    let eos = 999;
    let result = generate_with_speculation(
        &prompt,
        10,
        eos,
        |_tok, _pos| {
            let mut l = vec![0.0f32; 100];
            l[42] = 1.0;
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");
    assert_eq!(result.len(), 10);
}

#[test]
fn generate_does_not_overrun_budget_on_multi_token_draft() {
    // Regression for #242: the speculative branch committed every accepted
    // draft token (plus a rejection-fallback token) without re-checking the
    // budget, so a multi-token draft could return more than `max_new_tokens`.
    //
    // The suffix [7,8,9] recurs at pos 0, so `speculate(&prompt)` drafts
    // prompt[3..7] = [0,7,8,9] on the very first step. The forward_fn below
    // makes the target agree with all four (accepted == 4), so a budget of 1
    // would have returned 4 tokens before the fix.
    let prompt = vec![7u32, 8, 9, 0, 7, 8, 9];
    let eos = 999u32;
    for max_new in 1..=3 {
        let out = generate_with_speculation(
            &prompt,
            max_new,
            eos,
            |tok: u32, _pos: usize| match tok {
                0 => logits_with_argmax(1000, 7),
                7 => logits_with_argmax(1000, 8),
                8 => logits_with_argmax(1000, 9),
                _ => logits_with_argmax(1000, 5),
            },
            5,
            4,
        )
        .expect("non-empty prompt must generate successfully");
        assert!(
            out.len() <= max_new,
            "budget {max_new} overrun: produced {} tokens {:?}",
            out.len(),
            out
        );
        assert_eq!(out.len(), max_new, "should fill the budget exactly");
    }
}

#[test]
fn generate_eos_in_draft_stops_early() {
    // Prompt: [1, 2, 3, 99, 5] where eos=99.
    //
    // Iter 1: draft=[] (no 2-gram match from tail [..,5]); verify_draft(5,[],5,fwd).
    //   call 1: fwd(5, 5) → 2.  Emit 2.  all_tokens=[1,2,3,99,5,2], pos=6.
    // Iter 2: draft=[] ([5,2] not in prompt); verify_draft(2,[],6,fwd).
    //   call 2: fwd(2, 6) → 3.  Emit 3.  all_tokens=[...,2,3], pos=7.
    // Iter 3: 2-gram [2,3] matches prompt[1..3] → draft=[99,5].
    //   verify_draft(3, [99,5], 7, fwd):
    //     call 3: fwd(3, 7) → 99 == draft[0] → accepted=1, input=99
    //     call 4: fwd(99, 8) → 0  != draft[1]=5 → break
    //   Commit draft[..1]=[99]: 99==eos → return [2,3].
    let prompt = vec![1u32, 2, 3, 99, 5];
    let eos = 99;
    let mut call = 0;
    let result = generate_with_speculation(
        &prompt,
        20,
        eos,
        |_tok, _pos| {
            call += 1;
            let mut l = vec![0.0f32; 100];
            match call {
                1 => l[2] = 1.0,  // verify_draft(5,[],5): bonus → emit 2
                2 => l[3] = 1.0,  // verify_draft(2,[],6): bonus → emit 3
                3 => l[99] = 1.0, // verify_draft(3,[99,5],7): fwd(3,7)→99 matches draft[0]
                4 => l[0] = 1.0,  // verify_draft cont: fwd(99,8)→0 != draft[1]=5 → reject
                _ => l[42] = 1.0,
            }
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");

    // [2, 3] emitted normally, then draft[0]=99=EOS terminates generation
    assert_eq!(result, vec![2, 3]);
}

#[test]
fn generate_with_speculation_issue_243_regression() {
    // Regression: prompt=[1,1,1], greedy always picks 0 from input 1.
    // Speculation drafts [1,...] from 2-gram [1,1], but the model rejects
    // draft[0]=1 on the very first call (greedy choice is 0). The fix
    // ensures verify_draft feeds current_token first, so draft[0] is gated
    // by an actual forward pass rather than admitted unconditionally.
    let prompt = vec![1u32, 1, 1];
    let eos = 99;
    let result = generate_with_speculation(
        &prompt,
        1,
        eos,
        |_tok, _pos| {
            // Greedy choice is always 0 regardless of input.
            let mut l = vec![0.0f32; 10];
            l[0] = 1.0;
            l
        },
        5,
        4,
    )
    .expect("non-empty prompt must generate successfully");
    // Greedy emits [0]; the old buggy code emitted [1] (draft committed unverified).
    assert_eq!(result, vec![0]);
}

#[test]
fn generate_with_speculation_matches_greedy() {
    // Verifies speculative decoding is token-identical to plain greedy.
    //
    // forward_fn: next token = (input_token + 1) % 10 (pure, position-independent).
    // Prompt repeats the cycle twice to guarantee 2-gram matches fire during
    // generation, exercising the speculation path on every iteration after
    // the first few tokens.
    let prompt: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1];
    let eos = 50u32; // unreachable in a 10-token cycle
    let max_new = 10usize;

    // Reference greedy: feed last token, argmax, repeat.
    let greedy_fwd = |tok: u32, _pos: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; 60];
        l[((tok + 1) % 10) as usize] = 1.0;
        l
    };
    let mut ref_tokens = prompt.clone();
    let mut greedy_out = Vec::with_capacity(max_new);
    for ref_pos in (prompt.len()..).take(max_new) {
        let logits = greedy_fwd(*ref_tokens.last().unwrap(), ref_pos);
        let t = argmax(&logits) as u32;
        if t == eos {
            break;
        }
        greedy_out.push(t);
        ref_tokens.push(t);
    }

    // Speculative path — identical forward logic, independent closure.
    let spec_fwd = |tok: u32, _pos: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; 60];
        l[((tok + 1) % 10) as usize] = 1.0;
        l
    };
    let spec_out = generate_with_speculation(&prompt, max_new, eos, spec_fwd, 5, 4)
        .expect("non-empty prompt must generate successfully");

    assert_eq!(
        spec_out, greedy_out,
        "speculative output diverged from greedy: spec={spec_out:?} greedy={greedy_out:?}"
    );
}

#[test]
fn generate_with_speculation_matches_greedy_with_ties() {
    // Regression guard for #280 (last-wins tie-break divergence).
    //
    // The existing `generate_with_speculation_matches_greedy` uses one-hot logits
    // so the top logit is never tied.  This test places TWO entries at the same
    // maximum value so `argmax`'s first-wins tie-break is exercised on every step.
    // Both greedy and speculative paths must resolve the tie identically (lower
    // token-id wins) and produce the same output sequence.
    //
    // forward_fn: tokens 2 and 5 share the peak (first-wins → always 2).
    // Prompt repeats the cycle twice so the 2-gram speculator fires.
    let prompt: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1];
    let eos = 50u32;
    let max_new = 10usize;

    // The tied forward: token 2 and token 5 are both at the peak; argmax returns
    // the first (lower) index, so greedy always emits 2 regardless of input token.
    let tied_fwd = |_tok: u32, _pos: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; 60];
        l[2] = 1.0; // first tied winner — argmax must return this
        l[5] = 1.0; // second tied entry at the same value
        l
    };

    // Reference greedy: identical logic, independent closure.
    let greedy_fwd = |_tok: u32, _pos: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; 60];
        l[2] = 1.0;
        l[5] = 1.0;
        l
    };
    let mut ref_tokens = prompt.clone();
    let mut greedy_out = Vec::with_capacity(max_new);
    for ref_pos in (prompt.len()..).take(max_new) {
        let logits = greedy_fwd(*ref_tokens.last().unwrap(), ref_pos);
        let t = argmax(&logits) as u32;
        if t == eos {
            break;
        }
        greedy_out.push(t);
        ref_tokens.push(t);
    }

    let spec_out = generate_with_speculation(&prompt, max_new, eos, tied_fwd, 5, 4)
        .expect("non-empty prompt must generate successfully");

    assert_eq!(
        spec_out, greedy_out,
        "speculative output diverged from greedy under tie-break: spec={spec_out:?} greedy={greedy_out:?}"
    );
}

#[test]
fn stateful_speculation_starts_from_prefill_and_forwards_final_budget_token() {
    let prompt = vec![4u32, 5];
    let initial_logits = logits_with_argmax(16, 7);
    let mut calls = Vec::new();

    let output = generate_with_speculation_from_prefill(
        &prompt,
        2,
        15,
        initial_logits,
        |token, position| {
            calls.push((token, position));
            Ok(logits_with_argmax(16, 8))
        },
        5,
        4,
    )
    .expect("prefill-seeded generation must succeed");

    assert_eq!(output, vec![7, 8]);
    assert_eq!(
        calls,
        vec![(7, prompt.len()), (8, prompt.len() + 1)],
        "the callback must receive generated tokens at their true positions, \
             including the final budget token"
    );
}

#[test]
fn stateful_speculation_verifies_draft_against_prefill_logits() {
    let prompt = vec![7u32, 8, 9, 0, 7, 8, 9];
    let mut next = [7u32, 8, 9, 1].into_iter();
    let mut calls = Vec::new();

    let output = generate_with_speculation_from_prefill(
        &prompt,
        4,
        31,
        logits_with_argmax(32, 0),
        |token, position| {
            calls.push((token, position));
            Ok(logits_with_argmax(
                32,
                next.next().expect("each emitted token must be forwarded"),
            ))
        },
        5,
        4,
    )
    .expect("matching draft must succeed");

    assert_eq!(output, vec![0, 7, 8, 9]);
    assert_eq!(
        calls,
        vec![(0, 7), (7, 8), (8, 9), (9, 10)],
        "each accepted draft token must advance live state exactly once"
    );
}

#[test]
fn stateful_speculation_handles_draft_mismatch_and_eos_during_verification() {
    let prompt = vec![7u32, 8, 9, 0, 7, 8, 9];

    let mut mismatch_calls = Vec::new();
    let mismatch = generate_with_speculation_from_prefill(
        &prompt,
        1,
        31,
        logits_with_argmax(32, 6),
        |token, position| {
            mismatch_calls.push((token, position));
            Ok(logits_with_argmax(32, 5))
        },
        5,
        4,
    )
    .expect("a rejected draft must emit and forward the model choice");
    assert_eq!(mismatch, vec![6]);
    assert_eq!(mismatch_calls, vec![(6, prompt.len())]);

    let eos = 3u32;
    let mut eos_calls = Vec::new();
    let stopped = generate_with_speculation_from_prefill(
        &prompt,
        4,
        eos,
        logits_with_argmax(32, 0),
        |token, position| {
            eos_calls.push((token, position));
            Ok(logits_with_argmax(32, eos))
        },
        5,
        4,
    )
    .expect("EOS during draft verification is a successful stop");
    assert_eq!(stopped, vec![0]);
    assert_eq!(
        eos_calls,
        vec![(0, prompt.len())],
        "EOS must not be returned or forwarded"
    );
}

#[test]
fn stateful_speculation_stops_before_forwarding_eos_and_propagates_step_errors() {
    let eos = 3u32;
    let no_forward = generate_with_speculation_from_prefill(
        &[1, 2],
        4,
        eos,
        logits_with_argmax(8, eos),
        |_token, _position| panic!("EOS selected from prefill logits must not be forwarded"),
        5,
        4,
    )
    .expect("EOS is a successful stop");
    assert!(no_forward.is_empty());

    let error = generate_with_speculation_from_prefill(
        &[1, 2],
        1,
        eos,
        logits_with_argmax(8, 4),
        |_token, _position| {
            Err(crate::error::InferenceError::Inference(
                "synthetic step failure".into(),
            ))
        },
        5,
        4,
    );
    assert!(matches!(
        error,
        Err(crate::error::InferenceError::Inference(ref message))
            if message == "synthetic step failure"
    ));
}

// ── rejection_sample_draft tests ─────────────────────────────────────────

/// Build uniform logits over `vocab` tokens — after softmax every token
/// has probability 1/vocab.
fn uniform_logits(vocab: usize) -> Vec<f32> {
    vec![1.0f32; vocab]
}

/// Build peaked logits where token `tok` dominates.
fn peaked_logits(vocab: usize, tok: usize, peak: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; vocab];
    v[tok] = peak;
    v
}

#[test]
fn rejection_empty_draft_returns_no_bonus() {
    let result =
        rejection_sample_draft(&[], &[], &uniform_logits(8), &[], false, 1.0, Some(42)).unwrap();
    assert_eq!(result.accepted_count, 0);
    assert!(result.accepted_tokens.is_empty());
    assert_eq!(result.bonus_token, None);
    assert!(!result.had_rejection);
}

#[test]
fn rejection_mismatched_lengths_returns_error() {
    let logits = vec![peaked_logits(10, 3, 5.0)];
    let initial = uniform_logits(10);
    let err = rejection_sample_draft(&[3], &logits, &initial, &[], false, 1.0, Some(1));
    assert!(err.is_err());
}

#[test]
fn rejection_probabilistic_rejects_temperature_with_overflowing_reciprocal() {
    // 1e-40 is a positive finite subnormal whose reciprocal is +inf, which
    // would turn the centered maximum into `0.0 * inf = NaN` in the softmax.
    const VOCAB: usize = 4;
    let draft = vec![peaked_logits(VOCAB, 1, 3.0)];
    let target = vec![peaked_logits(VOCAB, 2, 5.0)];
    let initial = peaked_logits(VOCAB, 1, 4.0);

    let err = rejection_sample_draft(&[1u32], &draft, &initial, &target, false, 1e-40, Some(7))
        .expect_err("temperature with a non-finite reciprocal must be refused");
    match err {
        crate::error::InferenceError::InvalidInput(msg) => {
            assert!(
                msg.contains("1e-40"),
                "message must name the temperature: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // Control: a tiny temperature whose reciprocal is still finite is accepted.
    let ok = rejection_sample_draft(&[1u32], &draft, &initial, &target, false, 1e-30, Some(7));
    assert!(ok.is_ok(), "finite reciprocal must be accepted, got {ok:?}");
}

#[test]
fn rejection_probabilistic_rejects_nonfinite_logits() {
    // The probabilistic path turns every logit row into a softmax. A NaN or
    // +inf logit poisons softmax_into (sum=NaN -> the `sum > 0.0` guard skips
    // normalization -> un-normalized/NaN mass corrupts the accept test), so
    // the verifier must fail closed with InvalidInput rather than silently
    // accepting an invalid distribution. (#327/#335/#361 bug class.)
    // Each case mutates exactly ONE logit row; reverting the guard makes the
    // call return Ok(...) instead of Err, so this is mutation-sensitive.
    const VOCAB: usize = 4;

    let mut nan_target = uniform_logits(VOCAB);
    nan_target[1] = f32::NAN;
    let mut pos_inf_target = uniform_logits(VOCAB);
    pos_inf_target[2] = f32::INFINITY;
    let all_neg_inf = vec![f32::NEG_INFINITY; VOCAB];

    // (label, initial_target, draft_logits[0])
    let cases: [(&str, Vec<f32>, Vec<f32>); 4] = [
        ("nan_initial_target", nan_target, uniform_logits(VOCAB)),
        (
            "pos_inf_initial_target",
            pos_inf_target,
            uniform_logits(VOCAB),
        ),
        ("nan_draft", uniform_logits(VOCAB), {
            let mut d = uniform_logits(VOCAB);
            d[0] = f32::NAN;
            d
        }),
        (
            "all_neg_inf_initial_target",
            all_neg_inf,
            uniform_logits(VOCAB),
        ),
    ];

    for (label, initial_target, draft) in cases {
        let err = rejection_sample_draft(
            &[1u32],
            &[draft],
            &initial_target,
            &[peaked_logits(VOCAB, 2, 5.0)],
            false,
            1.0,
            Some(7),
        )
        .expect_err(&format!(
            "{label}: probabilistic verifier must reject non-finite logits"
        ));
        assert!(
            matches!(err, crate::error::InferenceError::InvalidInput(_)),
            "{label}: expected InvalidInput, got {err:?}"
        );
    }

    // The same non-finite inputs are tolerated by the GREEDY path (argmax is
    // NaN-robust): the guard must NOT regress greedy callers.
    let mut nan_target = uniform_logits(VOCAB);
    nan_target[1] = f32::NAN;
    let greedy_ok = rejection_sample_draft(
        &[2u32],
        &[],
        &nan_target,
        &[peaked_logits(VOCAB, 2, 5.0)],
        true,
        1.0,
        Some(7),
    );
    assert!(
        greedy_ok.is_ok(),
        "greedy path must remain robust to non-finite logits, got {greedy_ok:?}"
    );

    // A legitimate -inf zero-probability sentinel WITH a finite peak must
    // still be accepted (this is how the other rejection tests build rows).
    let mut peaked_with_neg_inf = vec![f32::NEG_INFINITY; VOCAB];
    peaked_with_neg_inf[2] = 5.0;
    let valid = rejection_sample_draft(
        &[2u32],
        &[peaked_with_neg_inf.clone()],
        &peaked_with_neg_inf,
        &[peaked_with_neg_inf.clone()],
        false,
        1.0,
        Some(7),
    );
    assert!(
        valid.is_ok(),
        "-inf with a finite peak is a valid distribution, got {valid:?}"
    );
}

#[test]
fn rejection_identical_distributions_accept_all() {
    // When draft and target have exactly the same distribution (including the
    // distribution that predicts draft[0]), p(x)/q(x) = 1 everywhere so all
    // tokens accept.
    const VOCAB: usize = 8;
    let logits = peaked_logits(VOCAB, 2, 10.0);

    let draft_tokens = vec![2u32, 2, 2];
    let draft_logits = vec![logits.clone(), logits.clone(), logits.clone()];
    let target_logits = vec![logits.clone(), logits.clone(), logits.clone()];
    let initial_target = logits.clone();

    for seed in 0u64..20 {
        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            1.0,
            Some(seed),
        )
        .unwrap();
        assert_eq!(
            res.accepted_count, 3,
            "seed={seed}: identical distributions must accept all tokens"
        );
        assert_eq!(res.accepted_tokens, vec![2u32, 2, 2]);
        assert!(!res.had_rejection, "seed={seed}: no rejection expected");
    }
}

#[test]
fn rejection_known_rejection_samples_adjusted() {
    // Single draft slot: q peaks exactly at 0, p (== initial_target_logits) peaks
    // exactly at 1 → accept_prob = min(1, 0/1) = 0, adjusted distribution puts all
    // mass on token 1, so every bonus draw must be token 1.
    const VOCAB: usize = 4;
    let draft_tok = 0u32;
    let mut draft_logit = vec![f32::NEG_INFINITY; VOCAB];
    draft_logit[0] = 100.0;
    let mut target_logit = vec![f32::NEG_INFINITY; VOCAB];
    target_logit[1] = 100.0;

    let draft_tokens = vec![draft_tok];
    let draft_logits = vec![draft_logit];
    // For a single-draft scenario, `target_logits[0]` (bonus position) and the
    // verification distribution (`initial_target_logits`) carry the same peak.
    let target_logits = vec![target_logit.clone()];
    let initial_target = target_logit;

    let mut bonus_counts = [0u32; VOCAB];
    let trials = 40u64;
    for seed in 0u64..trials {
        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            1.0,
            Some(seed),
        )
        .unwrap();
        assert!(res.had_rejection, "seed={seed}: rejection must occur");
        assert_eq!(res.accepted_count, 0, "seed={seed}: no tokens accepted");
        if let Some(b) = res.bonus_token {
            bonus_counts[b as usize] += 1;
        }
    }
    assert_eq!(
        bonus_counts[1], trials as u32,
        "adjusted sampling must produce token 1 in all trials, got counts={bonus_counts:?}"
    );
}

#[test]
fn rejection_greedy_mode_accepts_matching_argmax() {
    // Verification convention:
    //   draft[0] verified against initial_target_logits
    //   draft[i>=1] verified against target_logits[i-1]
    //   target_logits[n-1] is the bonus on full accept.
    //
    // Setup: initial_target peaks at draft[0]=5 → draft[0] accepted. Then
    // target_logits[0] peaks at 9 ≠ draft[1]=7 → reject. bonus = 9.
    const VOCAB: usize = 10;
    let draft_tokens = vec![5u32, 7, 3];
    let initial_target = peaked_logits(VOCAB, 5, 5.0);
    let target_logits = vec![
        peaked_logits(VOCAB, 9, 5.0),
        peaked_logits(VOCAB, 2, 5.0),
        peaked_logits(VOCAB, 4, 5.0),
    ];
    let draft_logits = vec![
        uniform_logits(VOCAB),
        uniform_logits(VOCAB),
        uniform_logits(VOCAB),
    ];

    let res = rejection_sample_draft(
        &draft_tokens,
        &draft_logits,
        &initial_target,
        &target_logits,
        true,
        1.0,
        Some(1),
    )
    .unwrap();

    assert_eq!(res.accepted_count, 1, "only position 0 accepted");
    assert!(res.had_rejection);
    assert_eq!(
        res.bonus_token,
        Some(9),
        "bonus is argmax of target_logits[0]"
    );
}

#[test]
fn rejection_greedy_all_match() {
    // Greedy: draft[0]=3 matches initial_target=3; draft[1]=7 matches
    // target_logits[0]=7; draft[2]=2 matches target_logits[1]=2; bonus from
    // target_logits[2]=9.
    const VOCAB: usize = 10;
    let draft_tokens = vec![3u32, 7, 2];
    let initial_target = peaked_logits(VOCAB, 3, 5.0);
    let target_logits = vec![
        peaked_logits(VOCAB, 7, 5.0),
        peaked_logits(VOCAB, 2, 5.0),
        peaked_logits(VOCAB, 9, 5.0),
    ];
    let draft_logits = vec![
        uniform_logits(VOCAB),
        uniform_logits(VOCAB),
        uniform_logits(VOCAB),
    ];

    let res = rejection_sample_draft(
        &draft_tokens,
        &draft_logits,
        &initial_target,
        &target_logits,
        true,
        1.0,
        Some(7),
    )
    .unwrap();

    assert_eq!(res.accepted_count, 3);
    assert_eq!(res.accepted_tokens, vec![3u32, 7, 2]);
    assert!(!res.had_rejection);
    assert_eq!(res.bonus_token, Some(9));
}

#[test]
fn rejection_q_zero_causes_rejection() {
    // q(x) = 0 for the draft token → accept_prob = 0 → always reject. On
    // rejection, bonus comes from the position-0 target distribution (=
    // initial_target_logits).
    const VOCAB: usize = 4;
    let mut draft_logit = vec![f32::NEG_INFINITY; VOCAB];
    draft_logit[0] = 10.0; // q peaks at 0
    let target_logit = peaked_logits(VOCAB, 2, 10.0); // p peaks at 2

    let res = rejection_sample_draft(
        &[2u32],
        &[draft_logit],
        &target_logit,
        std::slice::from_ref(&target_logit),
        false,
        1.0,
        Some(99),
    )
    .unwrap();

    assert!(res.had_rejection);
    assert_eq!(
        res.bonus_token,
        Some(2),
        "bonus token from adjusted dist should be 2"
    );
}

#[test]
fn rejection_probabilistic_full_accept_has_bonus() {
    // Identical distributions at every position → all accepted, bonus appended.
    const VOCAB: usize = 8;
    let logit = peaked_logits(VOCAB, 5, 15.0);
    let draft_tokens = vec![5u32, 5];
    let draft_logits = vec![logit.clone(), logit.clone()];
    let target_logits = vec![logit.clone(), logit.clone()];
    let initial_target = logit.clone();

    let res = rejection_sample_draft(
        &draft_tokens,
        &draft_logits,
        &initial_target,
        &target_logits,
        false,
        1.0,
        Some(0),
    )
    .unwrap();

    assert_eq!(res.accepted_count, 2);
    assert!(!res.had_rejection);
    // Bonus is sampled from the strongly peaked target distribution → must be 5.
    assert_eq!(
        res.bonus_token,
        Some(5),
        "bonus from peaked target must be token 5"
    );
}

#[test]
fn softmax_into_sums_to_one() {
    let logits = vec![1.0f32, 2.0, 3.0, 0.5, -1.0];
    let mut out = vec![0.0f32; logits.len()];
    softmax_into(&logits, &mut out);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax must sum to 1.0, got {sum}"
    );
    for &p in &out {
        assert!(p >= 0.0, "all softmax outputs must be non-negative");
    }
}

#[test]
fn softmax_into_peaked_distribution() {
    // A very large logit gap should produce ≈1.0 for the argmax.
    let mut logits = vec![0.0f32; 10];
    logits[3] = 100.0;
    let mut out = vec![0.0f32; 10];
    softmax_into(&logits, &mut out);
    assert!(
        out[3] > 0.999,
        "heavily peaked logit should dominate softmax"
    );
}

#[test]
fn sample_adjusted_concentrates_on_target_mass() {
    // p = [0.9, 0.1], q = [0.1, 0.9]
    // adjusted = max(0, p-q) = [0.8, 0.0], renormalised = [1.0, 0.0]
    // All samples should return token 0.
    let p = vec![0.9f32, 0.1];
    let q = vec![0.1f32, 0.9];
    for seed in 0u64..20 {
        let mut rng = SpecRng::new(seed + 1);
        let r = rng.next_f32();
        let tok = sample_adjusted(&p, &q, r);
        assert_eq!(tok, 0, "seed={seed}: all mass on token 0");
    }
}

mod mtp_rejection;
