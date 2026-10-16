//! MTP rejection-sampling integration and the MtpVerifier f16 KV cache rollback regression.

use super::*;

// ── ADR-050 MTP rejection sampling integration tests ─────────────────────

#[test]
fn mtp_rejection_sampling_seeded_acceptance_accepts_all() {
    // Identical draft and target distributions → p(x)/q(x) == 1 everywhere → all accept.
    const VOCAB: usize = 8;
    let logits = peaked_logits(VOCAB, 2, 100.0);
    let draft_tokens = vec![2u32, 2, 2];
    let draft_logits = vec![logits.clone(), logits.clone(), logits.clone()];
    let target_logits = vec![logits.clone(), logits.clone(), logits.clone()];
    let initial_target = logits.clone();

    let res = rejection_sample_draft(
        &draft_tokens,
        &draft_logits,
        &initial_target,
        &target_logits,
        false,
        1.0,
        Some(7),
    )
    .unwrap();

    assert_eq!(res.accepted_count, 3);
    assert_eq!(res.accepted_tokens, vec![2u32, 2, 2]);
    assert!(!res.had_rejection);
    assert_eq!(
        res.bonus_token,
        Some(2),
        "bonus from peaked target must be token 2"
    );
}

/// #329 mutation-sensitive determinism test for `rejection_sample_draft`'s `Some(seed)`
/// path. Every one of the 5 draft positions is built so `p(x)/q(x) == 0.5` exactly (via
/// `ln`-space logits, so `softmax` recovers the constructed probabilities exactly): each
/// accept/reject decision is a live coin flip driven by the RNG, not forced by a
/// one-hot distribution like `mtp_rejection_sampling_seeded_acceptance_accepts_all`
/// above. If seed threading is ever reverted to `SpecRng::from_clock()`, the two calls
/// below draw from different clock-seeded streams and — with 5 independent ~50/50
/// decisions in play — will almost certainly diverge in accepted_count, accepted_tokens,
/// or bonus_token, so this test is expected to fail under that mutation.
#[test]
fn rejection_sample_draft_same_seed_reproduces_probabilistic_decisions() {
    const N: usize = 5;

    // q(0) = 0.6, p(0) = 0.3 -> p(0)/q(0) == 0.5 exactly. Logits are ln(probability),
    // so softmax_into recovers [0.6, 0.4] / [0.3, 0.7] exactly (mod f32 rounding).
    let draft_row = vec![0.6f32.ln(), 0.4f32.ln()];
    let target_row = vec![0.3f32.ln(), 0.7f32.ln()];
    let draft_tokens = vec![0u32; N];
    let draft_logits = vec![draft_row.clone(); N];
    let target_logits = vec![target_row.clone(); N];
    let initial_target = target_row;

    let run = || {
        rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            1.0,
            Some(2024),
        )
        .unwrap()
    };

    let first = run();
    let second = run();

    assert_eq!(
        first.accepted_count, second.accepted_count,
        "same seed must reproduce the same accepted_count"
    );
    assert_eq!(
        first.accepted_tokens, second.accepted_tokens,
        "same seed must reproduce the same accepted_tokens"
    );
    assert_eq!(
        first.bonus_token, second.bonus_token,
        "same seed must reproduce the same bonus_token"
    );
}

/// #388 regression: the probabilistic path must compute `p`/`q` under the
/// caller-provided `temperature` (ADR-050), not a hardcoded `T = 1.0`. Uses a
/// 2-token vocab where token 1's accept probability `min(1, p(x)/q(x))` is a
/// genuinely fractional (non-clamped) value at both `T = 1.0` and `T = 4.0`,
/// and — because temperature scaling changes softmax sharpness non-uniformly —
/// the two accept probabilities differ. Both runs share the same seed, so the
/// RNG's first draw `u` is identical between them; the test scans seeds for one
/// where `u` lands strictly between the two accept probabilities, producing a
/// different accept/reject decision. If `temperature` is ever ignored (silently
/// hardcoded to `1.0` downstream, the #388 bug), both calls compute the exact
/// same `accept_prob` for every seed and `accepted_count` can never diverge —
/// this test is expected to fail under that mutation.
#[test]
fn rejection_sample_draft_temperature_changes_acceptance_decision() {
    let draft_tokens = vec![1u32];
    let draft_logits = vec![vec![0.0f32, 2.0]];
    let target_logits = vec![vec![0.0f32, 0.5]];
    let initial_target = vec![0.0f32, 0.5];

    let run_at = |temperature: f32, seed: u64| {
        rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            temperature,
            Some(seed),
        )
        .unwrap()
    };

    // `SpecRng` is a raw xorshift64 seeded directly with `seed` and no avalanche,
    // so small sequential seeds (0, 1, 2, …) all produce a near-zero *first* draw —
    // and with a single draft token that first `next_f32()` is the decision-critical
    // `u`. Scanning raw 0..200 therefore only ever samples `u ≈ 0`, which accepts at
    // both temperatures and never lands in the accept-probability gap, so spread the
    // seed across the u64 range (golden-ratio multiply) to sample `u` uniformly.
    let mut saw_divergence = false;
    for k in 0u64..200 {
        let seed = k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let low_t = run_at(1.0, seed);
        let high_t = run_at(4.0, seed);
        if low_t.accepted_count != high_t.accepted_count {
            saw_divergence = true;
            break;
        }
    }
    assert!(
        saw_divergence,
        "varying temperature (T=1.0 vs T=4.0) with identical logits and seed must be \
             able to flip the accept/reject decision — otherwise temperature is not reaching \
             the p/q softmax (ADR-050 / #388)"
    );
}

/// #388 regression: `mtp_verify_draft_with_seed` must forward its `temperature`
/// argument through to `rejection_sample_draft`'s `p`/`q` softmax rather than a
/// caller reverting just the plumbing (e.g. hardcoding `1.0` at the internal call
/// site) while `rejection_sample_draft` itself stays correct. Reuses
/// `probe_half_accept_prob_scenario`, whose two drafted positions are constructed
/// with a genuinely fractional (non one-hot) accept probability of exactly 0.5 at
/// `T = 1.0` — a non-degenerate distribution whose accept probability generically
/// changes under a different temperature. Scans seeds for one where the two
/// temperatures' `accepted_count` diverge; if the wrapper drops `temperature` on
/// the floor, every seed's outcome is identical between the two calls and the test
/// fails.
#[test]
fn mtp_verify_draft_with_seed_threads_temperature() {
    let max_seq = 8usize;

    let run_at = |temperature: f32, seed: u64| {
        let s = probe_half_accept_prob_scenario();
        let mut target = s.target;
        let mut verifier =
            MtpVerifier::new(s.cfg, &s.weights, &s.embed, &s.lm_head, max_seq).unwrap();
        mtp_verify_draft_with_seed(
            &mut verifier,
            s.current_token_id,
            s.current_position,
            &s.prev_hidden,
            &s.initial_target,
            None,
            &mut target,
            temperature,
            Some(seed),
        )
        .unwrap()
    };

    let mut saw_divergence = false;
    for seed in 0u64..200 {
        let low_t = run_at(1.0, seed);
        let high_t = run_at(5.0, seed);
        if low_t.accepted_count != high_t.accepted_count {
            saw_divergence = true;
            break;
        }
    }
    assert!(
        saw_divergence,
        "varying temperature (T=1.0 vs T=5.0) through mtp_verify_draft_with_seed must be \
             able to flip at least one seed's accept/reject decision — otherwise the wrapper \
             is not forwarding temperature into rejection_sample_draft (#388)"
    );
}

/// #389 regression: `mtp_verify_draft` must expose `rs.bonus_token` as
/// `fallback_token` on full accept, not only on `had_rejection`. Drives the real
/// `mtp_verify_draft` (not the greedy `mtp_verify_precomputed_draft` helper, which
/// never calls `rejection_sample_draft`) so the mapping under test is exercised.
/// #329: `mtp_verify_draft` now seeds its RNG from `MTP_VERIFY_DEFAULT_SEED`, not the
/// clock, but this test builds the scenario so the outcome is forced regardless of the
/// draw anyway: the draft/target distributions at every drafted position are
/// bit-identical (accept_prob == 1.0 exactly), and the bonus distribution is one-hot
/// (probability 1.0 on a single token), so no seed can produce a different
/// accepted-count or bonus token.
#[test]
fn mtp_verify_draft_full_accept_exposes_bonus_token() {
    const BONUS_TOKEN: u32 = 6;

    let mut cfg = tiny_mtp_config();
    cfg.draft_length = 2;
    let weights = tiny_mtp_weights(&cfg);
    let vocab = cfg.vocab_size;
    let h = cfg.hidden_size;
    let max_seq = 8usize;

    let embed: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let current_token_id = 1u32;
    let current_position = 0usize;

    // Probe the deterministic draft tokens/logits on a throwaway verifier so the
    // constructed target distributions can be made bit-identical to them.
    let mut probe = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
    let draft = probe
        .draft_tokens_with_logits(current_token_id, current_position, &prev_hidden, None)
        .unwrap();
    assert_eq!(
        draft.tokens.len(),
        2,
        "draft_length=2 with no EOS must draft 2 tokens"
    );

    let mut bonus_logits = vec![f32::NEG_INFINITY; vocab];
    bonus_logits[BONUS_TOKEN as usize] = 0.0;

    let mut target = MockTargetVerifier {
        cache_pos: current_position + 1,
        // target_dist_for(0) == initial_target_logits (below) == draft.logits[0];
        // target_dist_for(1) == logits_by_step[0] == draft.logits[1]: both drafted
        // positions get p(x) == q(x) exactly, forcing acceptance regardless of seed.
        // logits_by_step[1] (the position after the last accepted draft) is one-hot,
        // forcing the sampled bonus to be BONUS_TOKEN regardless of seed.
        logits_by_step: vec![draft.logits[1].clone(), bonus_logits],
        calls: vec![],
    };

    let mut verifier = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();
    let result = mtp_verify_draft(
        &mut verifier,
        current_token_id,
        current_position,
        &prev_hidden,
        &draft.logits[0],
        None,
        &mut target,
        1.0,
    )
    .unwrap();

    assert_eq!(
        result.accepted_count, 2,
        "both drafted tokens must be accepted"
    );
    assert_eq!(result.accepted_tokens, draft.tokens);
    assert!(!result.stopped_by_eos);
    assert_eq!(
        result.fallback_token,
        Some(BONUS_TOKEN),
        "full-accept must expose the sampled bonus token as fallback_token"
    );
}

/// Target mock that supplies post-final-norm hidden states. `last` and `rows` stand in for
/// `last_hidden` and the second element of `verify_tokens_with_hidden`; `None` behaves like a
/// target that cannot supply them.
struct HiddenTargetVerifier {
    inner: MockTargetVerifier,
    last: Option<Vec<f32>>,
    rows: Option<Vec<Vec<f32>>>,
}

impl MtpTargetVerifier for HiddenTargetVerifier {
    fn cache_position(&self) -> usize {
        self.inner.cache_position()
    }
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        self.inner.rollback_cache_to(seq_len)
    }
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        self.inner.verify_tokens(tokens, start_pos)
    }
    fn last_hidden(&self) -> Option<Vec<f32>> {
        self.last.clone()
    }
    #[allow(clippy::type_complexity)]
    fn verify_tokens_with_hidden(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), crate::error::InferenceError> {
        Ok((
            self.inner.verify_tokens(tokens, start_pos)?,
            self.rows.clone(),
        ))
    }
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        Vec::new()
    }
    fn restore_gdn_states(&mut self, _snapshot: &crate::attention::gdn::GdnSnapshot) {}
}

const HS_TOKEN: u32 = 1;
const HS_POSITION: usize = 0;

struct HiddenScenario {
    cfg: MtpConfig,
    weights: MtpWeights,
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    hidden_predicting: Vec<f32>,
    draft: MtpDraft,
    eos: Option<u32>,
    max_seq: usize,
}

/// Probe a throwaway verifier for the deterministic draft so target distributions can be
/// built bit-identical to it (accept probability exactly 1.0 at every position).
/// `stop_after_first` makes the first drafted token the EOS token, so the draft is one token
/// long while `draft_length` stays at the minimum the loop accepts.
fn hidden_scenario(draft_length: usize, stop_after_first: bool, max_seq: usize) -> HiddenScenario {
    let mut cfg = tiny_mtp_config();
    cfg.draft_length = draft_length;
    let weights = tiny_mtp_weights(&cfg);
    let vocab = cfg.vocab_size;
    let h = cfg.hidden_size;
    let embed: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();
    let hidden_predicting: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let probe_draft = |eos: Option<u32>| {
        let mut probe = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
        probe
            .draft_tokens_with_logits(HS_TOKEN, HS_POSITION, &hidden_predicting, eos)
            .unwrap()
    };
    let mut draft = probe_draft(None);
    let mut eos = None;
    if stop_after_first {
        eos = Some(draft.tokens[0]);
        draft = probe_draft(eos);
        assert_eq!(
            draft.tokens.len(),
            1,
            "an EOS first draft must stop the draft"
        );
    } else {
        assert_eq!(draft.tokens.len(), draft_length);
    }
    HiddenScenario {
        cfg,
        weights,
        embed,
        lm_head,
        hidden_predicting,
        draft,
        eos,
        max_seq,
    }
}

fn one_hot_logits(vocab: usize, token: u32) -> Vec<f32> {
    let mut l = vec![f32::NEG_INFINITY; vocab];
    l[token as usize] = 0.0;
    l
}

/// Target logits for the verify call when exactly `accepted` draft tokens agree with the
/// target: p == q for every agreeing position, a one-hot disagreement at the first rejected
/// one, and a one-hot bonus after the last position.
fn hs_step_logits(s: &HiddenScenario, accepted: usize) -> Vec<Vec<f32>> {
    let vocab = s.cfg.vocab_size;
    let mut l: Vec<Vec<f32>> = s.draft.logits[1..].to_vec();
    l.push(one_hot_logits(vocab, 6));
    if accepted >= 1 && accepted < s.draft.tokens.len() {
        let mismatch = (s.draft.tokens[accepted] + 1) % vocab as u32;
        l[accepted - 1] = one_hot_logits(vocab, mismatch);
    }
    l
}

/// Target logits for the pending token: they agree with the first draft token unless
/// `accepted` is zero.
fn hs_initial_logits(s: &HiddenScenario, accepted: usize) -> Vec<f32> {
    if accepted == 0 {
        let vocab = s.cfg.vocab_size;
        one_hot_logits(vocab, (s.draft.tokens[0] + 1) % vocab as u32)
    } else {
        s.draft.logits[0].clone()
    }
}

/// One distinct, non-trivial hidden row per index.
fn distinct_hidden_rows(n: usize, hidden: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..hidden)
                .map(|j| ((i * hidden + j) as f32 * 0.7 + 0.3).sin() + 0.5 * i as f32)
                .collect()
        })
        .collect()
}

fn hidden_target(
    logits_by_step: Vec<Vec<f32>>,
    last: Option<Vec<f32>>,
    rows: Option<Vec<Vec<f32>>>,
) -> HiddenTargetVerifier {
    HiddenTargetVerifier {
        inner: MockTargetVerifier {
            cache_pos: HS_POSITION + 1,
            logits_by_step,
            calls: vec![],
        },
        last,
        rows,
    }
}

fn hs_run<'a, T: MtpTargetVerifier>(
    s: &'a HiddenScenario,
    initial_target_logits: &[f32],
    target: &mut T,
) -> Result<(MtpVerifyResult, MtpVerifier<'a>), crate::error::InferenceError> {
    let mut verifier =
        MtpVerifier::new(s.cfg.clone(), &s.weights, &s.embed, &s.lm_head, s.max_seq).unwrap();
    let result = mtp_verify_draft(
        &mut verifier,
        HS_TOKEN,
        HS_POSITION,
        &s.hidden_predicting,
        initial_target_logits,
        s.eos,
        target,
        1.0,
    )?;
    Ok((result, verifier))
}

fn cache_kv_row(v: &MtpVerifier<'_>, row: usize) -> (Vec<half::f16>, Vec<half::f16>) {
    let kv_dim = v.config.num_key_value_heads * v.config.head_dim;
    (
        v.cache.k_buffer(0)[row * kv_dim..(row + 1) * kv_dim].to_vec(),
        v.cache.v_buffer(0)[row * kv_dim..(row + 1) * kv_dim].to_vec(),
    )
}

/// The K/V row a full `forward_one` step writes for (`token`, `hidden`) at `position` on a
/// fresh verifier: the independent computation an appended row must equal.
fn reference_kv_row(
    s: &HiddenScenario,
    token: u32,
    position: usize,
    hidden: &[f32],
) -> (Vec<half::f16>, Vec<half::f16>) {
    let mut v =
        MtpVerifier::new(s.cfg.clone(), &s.weights, &s.embed, &s.lm_head, s.max_seq).unwrap();
    v.forward_one(token, position, hidden).unwrap();
    cache_kv_row(&v, 0)
}

/// The MTP cache holds the row for the current token (built from the caller's hidden state)
/// followed by one row per accepted draft token, each built by a fresh `forward_one` from the
/// token's own position and the hidden state that predicted it.
fn assert_target_hidden_rows(
    s: &HiddenScenario,
    v: &MtpVerifier<'_>,
    accepted: &[u32],
    last: &[f32],
    rows: &[Vec<f32>],
) {
    assert_eq!(
        v.cache.seq_len(),
        1 + accepted.len(),
        "cursor must be the current token's row plus one row per accepted draft token"
    );
    assert_eq!(
        cache_kv_row(v, 0),
        reference_kv_row(s, HS_TOKEN, HS_POSITION, &s.hidden_predicting),
        "row for the current token keeps the caller's hidden state"
    );
    for (i, &token) in accepted.iter().enumerate() {
        let k = i + 1;
        let predictor = if k == 1 { last } else { rows[k - 2].as_slice() };
        assert_eq!(
            cache_kv_row(v, k),
            reference_kv_row(s, token, HS_POSITION + k, predictor),
            "row for draft token {k}"
        );
    }
}

#[test]
fn mtp_verify_draft_target_hidden_full_accept_rewrites_rows() {
    let s = hidden_scenario(3, false, 8);
    let all = distinct_hidden_rows(4, s.cfg.hidden_size);
    let rows = all[..3].to_vec();
    let last = all[3].clone();
    let mut target = hidden_target(
        hs_step_logits(&s, 3),
        Some(last.clone()),
        Some(rows.clone()),
    );

    let (result, v) = hs_run(&s, &hs_initial_logits(&s, 3), &mut target).unwrap();

    assert_eq!(result.accepted_count, 3);
    assert_eq!(target.inner.cache_pos, HS_POSITION + 1 + 3);
    assert_target_hidden_rows(&s, &v, &s.draft.tokens, &last, &rows);

    // The rewrite is observable: drafting built rows 1 and 2 from the draft head's own
    // hidden state, which differs from the target's.
    let mut plain = MockTargetVerifier {
        cache_pos: HS_POSITION + 1,
        logits_by_step: hs_step_logits(&s, 3),
        calls: vec![],
    };
    let (_, plain_v) = hs_run(&s, &hs_initial_logits(&s, 3), &mut plain).unwrap();
    for k in 1..3 {
        assert_ne!(
            cache_kv_row(&v, k),
            cache_kv_row(&plain_v, k),
            "row {k} must come from the target's hidden state, not the drafted one"
        );
    }
}

#[test]
fn mtp_verify_draft_target_hidden_partial_accept_rewrites_prefix_rows() {
    let s = hidden_scenario(3, false, 8);
    let all = distinct_hidden_rows(4, s.cfg.hidden_size);
    let rows = all[..3].to_vec();
    let last = all[3].clone();
    let mut target = hidden_target(
        hs_step_logits(&s, 2),
        Some(last.clone()),
        Some(rows.clone()),
    );

    let (result, v) = hs_run(&s, &hs_initial_logits(&s, 2), &mut target).unwrap();

    assert_eq!(result.accepted_count, 2);
    assert_eq!(target.inner.cache_pos, HS_POSITION + 1 + 2);
    assert_target_hidden_rows(&s, &v, &s.draft.tokens[..2], &last, &rows);
}

#[test]
fn mtp_verify_draft_target_hidden_reject_keeps_current_token_row() {
    let s = hidden_scenario(3, false, 8);
    let all = distinct_hidden_rows(4, s.cfg.hidden_size);
    let rows = all[..3].to_vec();
    let last = all[3].clone();
    let mut target = hidden_target(
        hs_step_logits(&s, 0),
        Some(last.clone()),
        Some(rows.clone()),
    );

    let (result, v) = hs_run(&s, &hs_initial_logits(&s, 0), &mut target).unwrap();

    assert_eq!(result.accepted_count, 0);
    assert_eq!(target.inner.cache_pos, HS_POSITION + 1);
    assert_target_hidden_rows(&s, &v, &[], &last, &rows);

    // Before, a reject also discarded the current token's own row.
    let mut plain = MockTargetVerifier {
        cache_pos: HS_POSITION + 1,
        logits_by_step: hs_step_logits(&s, 0),
        calls: vec![],
    };
    let (_, plain_v) = hs_run(&s, &hs_initial_logits(&s, 0), &mut plain).unwrap();
    assert_eq!(plain_v.cache.seq_len(), 0);
}

#[test]
fn mtp_verify_draft_target_hidden_first_draft_row_uses_last_hidden() {
    // One drafted token: its row cannot come from a verify row (the verify call starts at
    // that token), so the only correct source is the hidden state after the current token.
    let s = hidden_scenario(2, true, 8);
    let all = distinct_hidden_rows(2, s.cfg.hidden_size);
    let rows = all[..1].to_vec();
    let last = all[1].clone();
    let mut target = hidden_target(
        hs_step_logits(&s, 1),
        Some(last.clone()),
        Some(rows.clone()),
    );

    let (result, v) = hs_run(&s, &hs_initial_logits(&s, 1), &mut target).unwrap();

    assert_eq!(result.accepted_count, 1);
    assert_target_hidden_rows(&s, &v, &s.draft.tokens[..1], &last, &rows);
    let token = s.draft.tokens[0];
    let right = reference_kv_row(&s, token, HS_POSITION + 1, &last);
    for wrong in [&s.hidden_predicting, &rows[0]] {
        assert_ne!(
            right,
            reference_kv_row(&s, token, HS_POSITION + 1, wrong),
            "the scenario must make a wrong pairing observable"
        );
    }
}

#[test]
fn mtp_verify_draft_target_hidden_missing_input_leaves_cache_as_drafted() {
    let s = hidden_scenario(3, false, 8);
    let all = distinct_hidden_rows(4, s.cfg.hidden_size);
    let rows = all[..3].to_vec();
    let last = all[3].clone();

    for accepted in [3usize, 2, 0] {
        // The default trait methods: a target that only implements `verify_tokens`.
        let mut plain = MockTargetVerifier {
            cache_pos: HS_POSITION + 1,
            logits_by_step: hs_step_logits(&s, accepted),
            calls: vec![],
        };
        let (plain_r, plain_v) = hs_run(&s, &hs_initial_logits(&s, accepted), &mut plain).unwrap();
        assert_eq!(plain_r.accepted_count, accepted);
        assert_eq!(
            plain_v.cache.seq_len(),
            accepted,
            "cursor ends at the drafted rows of the accepted prefix"
        );

        let arms = [
            (Some(last.clone()), None),
            (None, Some(rows.clone())),
            (None, None),
        ];
        for (last_hidden, hidden_rows) in arms {
            let mut target = hidden_target(hs_step_logits(&s, accepted), last_hidden, hidden_rows);
            let (r, v) = hs_run(&s, &hs_initial_logits(&s, accepted), &mut target).unwrap();
            assert_eq!(r, plain_r, "accepted={accepted}");
            assert_eq!(v.cache.seq_len(), accepted, "accepted={accepted}");
            for k in 0..accepted {
                assert_eq!(cache_kv_row(&v, k), cache_kv_row(&plain_v, k));
            }
        }
    }
}

#[test]
fn mtp_verify_draft_target_hidden_row_count_mismatch_errs_and_restores_target() {
    let s = hidden_scenario(3, false, 8);
    let all = distinct_hidden_rows(3, s.cfg.hidden_size);
    let mut target = hidden_target(
        hs_step_logits(&s, 3),
        Some(all[2].clone()),
        Some(all[..2].to_vec()),
    );

    let err = hs_run(&s, &hs_initial_logits(&s, 3), &mut target).err();

    assert!(err.is_some(), "a short row set is a contract violation");
    assert_eq!(
        target.inner.cache_pos,
        HS_POSITION + 1,
        "target cache restored to its pre-call position"
    );
}

#[test]
fn mtp_verify_draft_target_hidden_full_accept_at_mtp_capacity_keeps_cursor() {
    // The MTP cache holds exactly the drafted rows, so there is no room for the rewrite's
    // extra row. The call still succeeds and the cursor stops at capacity.
    let s = hidden_scenario(3, false, 3);
    let all = distinct_hidden_rows(4, s.cfg.hidden_size);
    let mut target = hidden_target(
        hs_step_logits(&s, 3),
        Some(all[3].clone()),
        Some(all[..3].to_vec()),
    );

    let (result, v) = hs_run(&s, &hs_initial_logits(&s, 3), &mut target).unwrap();

    assert_eq!(result.accepted_count, 3);
    assert_eq!(v.cache.seq_len(), 3);
}

/// Rescale `softmax(base_logits)[token]` by `ratio` (renormalised) and return the
/// resulting logits. Used to build MTP verify scenarios where a drafted token's accept
/// probability `p(x)/q(x)` is deliberately not 0 or 1, so the accept/reject decision is
/// actually driven by the RNG draw instead of being forced by construction.
///
/// Derivation: shifting only `base_logits[token]` by an additive constant `a` and
/// renormalising gives `new_p(token) = q(token) * exp(a) / (1 + q(token) * (exp(a) - 1))`.
/// Solving for `a` such that `new_p(token) / q(token) == ratio` yields
/// `a = ln(ratio * (1 - q(token)) / (1 - ratio * q(token)))`.
fn logits_with_scaled_prob(base_logits: &[f32], token: usize, ratio: f32) -> Vec<f32> {
    let mut q = vec![0.0f32; base_logits.len()];
    softmax_into(base_logits, &mut q);
    let qx = f64::from(q[token]);
    let ratio = f64::from(ratio);
    let denom = 1.0 - ratio * qx;
    assert!(
        denom > 0.0,
        "ratio={ratio} too large for q(token)={qx} to be achievable by rescaling"
    );
    let shift = ((ratio * (1.0 - qx)) / denom).ln();
    let mut out = base_logits.to_vec();
    out[token] += shift as f32;
    out
}

/// Bundles the pieces needed to run [`mtp_verify_draft`]/[`mtp_verify_draft_with_seed`]
/// against a scenario where every drafted position's accept probability is exactly 0.5,
/// built by [`probe_half_accept_prob_scenario`].
struct HalfAcceptProbScenario {
    cfg: MtpConfig,
    weights: MtpWeights,
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    prev_hidden: Vec<f32>,
    current_token_id: u32,
    current_position: usize,
    draft: MtpDraft,
    initial_target: Vec<f32>,
    target: MockTargetVerifier,
}

/// Shared scaffold for the #329 MTP-level determinism tests below: probes a real
/// `MtpVerifier` for its (deterministic) draft tokens/logits, then builds target
/// distributions where every drafted position's accept probability is exactly 0.5 —
/// via [`logits_with_scaled_prob`] — so each accept/reject decision genuinely depends
/// on the RNG draw rather than being forced one-hot. The bonus-token distribution stays
/// one-hot purely to keep assertions simple; it is not part of what makes this scenario
/// mutation-sensitive.
fn probe_half_accept_prob_scenario() -> HalfAcceptProbScenario {
    const BONUS_TOKEN: u32 = 5;

    let mut cfg = tiny_mtp_config();
    cfg.draft_length = 2;
    let weights = tiny_mtp_weights(&cfg);
    let vocab = cfg.vocab_size;
    let h = cfg.hidden_size;
    let max_seq = 8usize;

    let embed: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let current_token_id = 1u32;
    let current_position = 0usize;

    let mut probe = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
    let draft = probe
        .draft_tokens_with_logits(current_token_id, current_position, &prev_hidden, None)
        .unwrap();
    assert_eq!(
        draft.tokens.len(),
        2,
        "draft_length=2 with no EOS must draft 2 tokens"
    );

    let initial_target = logits_with_scaled_prob(&draft.logits[0], draft.tokens[0] as usize, 0.5);
    let pos1_target = logits_with_scaled_prob(&draft.logits[1], draft.tokens[1] as usize, 0.5);

    let mut bonus_logits = vec![f32::NEG_INFINITY; vocab];
    bonus_logits[BONUS_TOKEN as usize] = 0.0;

    let target = MockTargetVerifier {
        cache_pos: current_position + 1,
        logits_by_step: vec![pos1_target, bonus_logits],
        calls: vec![],
    };

    HalfAcceptProbScenario {
        cfg,
        weights,
        embed,
        lm_head,
        prev_hidden,
        current_token_id,
        current_position,
        draft,
        initial_target,
        target,
    }
}

/// #329 mutation-sensitive determinism test for `mtp_verify_draft_with_seed`. Both
/// drafted positions have accept probability exactly 0.5 (see
/// `probe_half_accept_prob_scenario`), so the outcome is a live RNG-driven decision, not
/// forced by construction. Each same-seed pair must produce identical results. A single
/// pair is a weak mutation detector: with 2 positions at 50/50 (rejection stops at the
/// first failed draw) a reverted clock-seeded path still matches by coincidence with
/// probability 0.5^2 + 0.25^2 + 0.25^2 = 0.375, so this runs 8 independent pairs —
/// a revert survives all of them with probability 0.375^8 ≈ 4e-4.
#[test]
fn mtp_verify_draft_with_seed_repeats_probabilistic_acceptance() {
    let max_seq = 8usize;

    let s1 = probe_half_accept_prob_scenario();
    let s2 = probe_half_accept_prob_scenario();
    assert_eq!(
        s1.draft.tokens, s2.draft.tokens,
        "probe must be deterministic"
    );

    let run_with_seed = |seed: u64| {
        let s = probe_half_accept_prob_scenario();
        let mut target = s.target;
        let mut verifier =
            MtpVerifier::new(s.cfg, &s.weights, &s.embed, &s.lm_head, max_seq).unwrap();
        mtp_verify_draft_with_seed(
            &mut verifier,
            s.current_token_id,
            s.current_position,
            &s.prev_hidden,
            &s.initial_target,
            None,
            &mut target,
            1.0,
            Some(seed),
        )
        .unwrap()
    };

    for seed in [
        4242u64,
        7,
        99,
        314_159,
        271_828,
        1_000_003,
        0xdead_beef,
        0x853c_49e6,
    ] {
        let first = run_with_seed(seed);
        let second = run_with_seed(seed);
        assert_eq!(
            first.accepted_count, second.accepted_count,
            "seed {seed}: same seed must reproduce the same accepted_count"
        );
        assert_eq!(
            first.accepted_tokens, second.accepted_tokens,
            "seed {seed}: same seed must reproduce the same accepted_tokens"
        );
        assert_eq!(
            first.fallback_token, second.fallback_token,
            "seed {seed}: same seed must reproduce the same fallback_token"
        );
    }
}

/// #329: locks the documented `seed=None` default behavior of the compatibility shim
/// `mtp_verify_draft` — it must be deterministic via `MTP_VERIFY_DEFAULT_SEED`, not
/// accidentally clock-seeded. Uses the same accept_prob == 0.5 scenario as
/// `mtp_verify_draft_with_seed_repeats_probabilistic_acceptance`, and the same
/// 8-independent-pairs repetition for the same reason: one 50/50 pair matches a
/// reverted clock-seeded path by coincidence with probability 0.375, eight pairs
/// push that below 4e-4.
#[test]
fn mtp_verify_draft_default_seed_is_deterministic() {
    let max_seq = 8usize;

    let run_default = || {
        let s = probe_half_accept_prob_scenario();
        let mut target = s.target;
        let mut verifier =
            MtpVerifier::new(s.cfg, &s.weights, &s.embed, &s.lm_head, max_seq).unwrap();
        mtp_verify_draft(
            &mut verifier,
            s.current_token_id,
            s.current_position,
            &s.prev_hidden,
            &s.initial_target,
            None,
            &mut target,
            1.0,
        )
        .unwrap()
    };

    for pair in 0..8 {
        let first = run_default();
        let second = run_default();
        assert_eq!(
            first.accepted_count, second.accepted_count,
            "pair {pair}: default seed=None must reproduce the same accepted_count"
        );
        assert_eq!(
            first.accepted_tokens, second.accepted_tokens,
            "pair {pair}: default seed=None must reproduce the same accepted_tokens"
        );
        assert_eq!(
            first.fallback_token, second.fallback_token,
            "pair {pair}: default seed=None must reproduce the same fallback_token"
        );
    }
}

/// #329: different explicit seeds are permitted to reach different outcomes — this test
/// only asserts that both seeded calls execute successfully to completion. It
/// deliberately does NOT assert the outputs differ: two distinct seeds are allowed to
/// coincidentally land on the same accept/reject decisions, so asserting inequality
/// would be flaky (the seed-evolution contract is "may differ", not "must differ").
#[test]
fn mtp_verify_draft_different_seed_is_allowed_to_vary() {
    let max_seq = 8usize;
    let s1 = probe_half_accept_prob_scenario();
    let mut target = s1.target;

    let mut verifier =
        MtpVerifier::new(s1.cfg, &s1.weights, &s1.embed, &s1.lm_head, max_seq).unwrap();
    let with_seed_a = mtp_verify_draft_with_seed(
        &mut verifier,
        s1.current_token_id,
        s1.current_position,
        &s1.prev_hidden,
        &s1.initial_target,
        None,
        &mut target,
        1.0,
        Some(1),
    );
    assert!(with_seed_a.is_ok(), "seed_a run must complete successfully");

    target.cache_pos = s1.current_position + 1;
    target.calls.clear();
    let s2 = probe_half_accept_prob_scenario();
    let mut verifier2 =
        MtpVerifier::new(s2.cfg, &s2.weights, &s2.embed, &s2.lm_head, max_seq).unwrap();
    let with_seed_b = mtp_verify_draft_with_seed(
        &mut verifier2,
        s2.current_token_id,
        s2.current_position,
        &s2.prev_hidden,
        &s2.initial_target,
        None,
        &mut target,
        1.0,
        Some(2),
    );
    assert!(with_seed_b.is_ok(), "seed_b run must complete successfully");
}

#[test]
fn mtp_rejection_sampling_seeded_rejection_samples_target_token() {
    // q peaks at 0, p peaks at 1 → accept_prob = 0 → certain rejection.
    // Adjusted distribution = max(0, p-q) has all mass at token 1 → bonus = Some(1).
    const VOCAB: usize = 4;
    let mut q_peak_0 = vec![f32::NEG_INFINITY; VOCAB];
    q_peak_0[0] = 100.0;
    let mut p_peak_1 = vec![f32::NEG_INFINITY; VOCAB];
    p_peak_1[1] = 100.0;

    let res = rejection_sample_draft(
        &[0u32],
        &[q_peak_0],
        &p_peak_1,
        &[p_peak_1.clone()],
        false,
        1.0,
        Some(13),
    )
    .unwrap();

    assert_eq!(res.accepted_count, 0);
    assert!(res.accepted_tokens.is_empty());
    assert!(res.had_rejection);
    assert_eq!(
        res.bonus_token,
        Some(1),
        "adjusted sampling must produce token 1"
    );
}

#[test]
fn mtp_rejection_sampling_partial_acceptance_rolls_back_gdn_to_accepted_prefix() {
    // 3-token draft: tokens [0, 1, 2].
    //   draft_logits  = [peak0, peak1, q_peak_2]  — draft model near-certain at each
    //   initial_target = peak0                     — p/q ≈ 1 at i=0 → accept token 0
    //   target_logits[0] = peak1                  — p/q ≈ 1 at i=1 → accept token 1
    //   target_logits[1] = p_peak_5               — p_peak_5[2]/q_peak_2[2] ≈ 0 → reject
    //   bonus = argmax(adjusted(p_peak_5, q_peak_2)) = 5
    const VOCAB: usize = 8;
    let peak0 = peaked_logits(VOCAB, 0, 100.0);
    let peak1 = peaked_logits(VOCAB, 1, 100.0);
    let q_peak_2 = peaked_logits(VOCAB, 2, 100.0);
    let p_peak_5 = peaked_logits(VOCAB, 5, 100.0);
    let bonus_peak_6 = peaked_logits(VOCAB, 6, 100.0);

    let rs = rejection_sample_draft(
        &[0u32, 1, 2],
        &[peak0.clone(), peak1.clone(), q_peak_2.clone()],
        &peak0,
        &[peak1.clone(), p_peak_5.clone(), bonus_peak_6.clone()],
        false,
        1.0,
        Some(0),
    )
    .unwrap();

    assert_eq!(rs.accepted_count, 2);
    assert_eq!(rs.accepted_tokens, vec![0u32, 1]);
    assert!(rs.had_rejection);
    assert_eq!(rs.bonus_token, Some(5));

    // Verify that callers (mtp_verify_draft) will correctly roll back cache to
    // target_start + accepted_count = target_start + 2.
    let target_start = 10usize;
    let mut restore_gdn_called = false;
    let mut mock = {
        struct GdnMock {
            cache_pos: usize,
            restore_called: *mut bool,
        }
        impl MtpTargetVerifier for GdnMock {
            fn cache_position(&self) -> usize {
                self.cache_pos
            }
            fn rollback_cache_to(
                &mut self,
                seq_len: usize,
            ) -> Result<(), crate::error::InferenceError> {
                self.cache_pos = seq_len;
                Ok(())
            }
            fn verify_tokens(
                &mut self,
                tokens: &[u32],
                _: usize,
            ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
                self.cache_pos += tokens.len();
                Ok(vec![])
            }
            fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
                Vec::new()
            }
            fn restore_gdn_states(&mut self, _: &crate::attention::gdn::GdnSnapshot) {
                // SAFETY: single-threaded test, pointer is to a local bool.
                unsafe {
                    *self.restore_called = true;
                }
            }
        }
        GdnMock {
            cache_pos: target_start,
            restore_called: &mut restore_gdn_called,
        }
    };

    // Simulate what mtp_verify_draft does: advance cache via verify_tokens, then roll back.
    mock.verify_tokens(&[0u32, 1, 2], target_start + 1).unwrap();
    mock.rollback_cache_to(target_start + rs.accepted_count)
        .unwrap();

    assert_eq!(
        mock.cache_pos,
        target_start + 2,
        "rollback must leave cache at target_start + accepted_count"
    );
    assert!(
        !restore_gdn_called,
        "partial accept must not call restore_gdn_states; rollback_cache_to handles GDN"
    );
}

#[test]
fn rejection_greedy_mode_regression_matches_existing_argmax_behavior() {
    // Regression guard: greedy=true path must still accept only argmax-matching tokens.
    // initial_target peaks at 5 → draft[0]=5 accepted.
    // target_logits[0] peaks at 9 ≠ draft[1]=7 → reject; bonus = 9.
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

    assert_eq!(
        res.accepted_count, 1,
        "only draft[0]=5 matches initial_target argmax=5"
    );
    assert!(res.had_rejection);
    assert_eq!(
        res.bonus_token,
        Some(9),
        "bonus is argmax of target_logits[0]"
    );
}

#[test]
fn rejection_greedy_one_draft_mtp_contract_empty_draft_logits() {
    // Locks the contract the greedy MTP loop (metal_qwen35::generate_greedy_mtp,
    // #237) depends on: a single draft token verified with `greedy=true` and an
    // EMPTY `draft_logits` slice (unused in greedy mode). The downstream loop reads
    //   accept → bonus = argmax(target_logits[0])  (verify_out.logits[1])
    //   reject → bonus = argmax(initial_target)     (verify_out.logits[0])
    // A regression that flips this call back to probabilistic (greedy=false,
    // clock-seeded) would break determinism and these argmax bonus identities.
    // `seed=None` with `greedy=true` must stay deterministic (no RNG is built).
    const VOCAB: usize = 16;
    let initial_target = peaked_logits(VOCAB, 4, 6.0); // target's pick at draft pos = 4
    let target_logits = vec![peaked_logits(VOCAB, 11, 6.0)]; // full-accept bonus = 11

    // Accept: draft token == initial_target argmax (4) → accept, bonus = 11.
    let accept = rejection_sample_draft(
        &[4u32],
        &[],
        &initial_target,
        &target_logits,
        true,
        1.0,
        None,
    )
    .unwrap();
    assert_eq!(
        accept.accepted_count, 1,
        "draft == initial argmax → accepted"
    );
    assert!(!accept.had_rejection);
    assert_eq!(
        accept.bonus_token,
        Some(11),
        "full-accept bonus is argmax(target_logits[0])"
    );

    // Reject: draft token (7) != initial_target argmax (4) → reject, bonus = 4.
    let reject = rejection_sample_draft(
        &[7u32],
        &[],
        &initial_target,
        &target_logits,
        true,
        1.0,
        None,
    )
    .unwrap();
    assert_eq!(
        reject.accepted_count, 0,
        "draft != initial argmax → rejected"
    );
    assert!(reject.had_rejection);
    assert_eq!(
        reject.bonus_token,
        Some(4),
        "rejection bonus is the target's own pick = argmax(initial_target)"
    );
}

// -----------------------------------------------------------------------
// MtpVerifier f16 KV cache rollback regression (Defect 4 fix, Option A).
//
// Exercises the real f16 FlatKVCache + dequant scratch path under rollback:
//   replay:  forward(1,0) → forward(2,1) → forward(3,2) → rollback(1)
//              → forward(7,1)
//   fresh:   forward(1,0) → forward(7,1)
//
// After both sequences the KV cache at positions 0 and 1 must be identical
// and the final logits must match, proving stale f16 data at positions 2/3
// does not contaminate the output after rollback.
// -----------------------------------------------------------------------

/// Build the smallest valid MtpConfig for rollback testing.
fn tiny_mtp_config() -> MtpConfig {
    MtpConfig {
        draft_length: 1,
        num_hidden_layers: 1,
        hidden_size: 4,
        vocab_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5, // rope_dim = 4*0.5 = 2
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 2,
        shared_expert_intermediate_size: 2,
        use_dedicated_embeddings: false,
    }
}

/// Build tiny MtpWeights for the given config with deterministic non-zero values.
fn tiny_mtp_weights(cfg: &MtpConfig) -> MtpWeights {
    let h = cfg.hidden_size;
    let q_proj_rows = 2 * cfg.num_attention_heads * cfg.head_dim;
    let kv_proj_rows = cfg.num_key_value_heads * cfg.head_dim;
    let moe_inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;
    let num_experts = cfg.num_experts;

    // Small but non-zero values: use position-based pattern to avoid symmetry.
    let fill = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| scale * ((i as f32 + 1.0) * 0.01).sin())
            .collect()
    };
    let ones = |n: usize| vec![1.0f32; n];

    let layer = MtpLayerWeights {
        input_layernorm: ones(h),
        post_attention_layernorm: ones(h),
        self_attn: MtpAttentionWeights {
            q_proj: fill(q_proj_rows * h, 0.1),
            k_proj: fill(kv_proj_rows * h, 0.1),
            v_proj: fill(kv_proj_rows * h, 0.1),
            o_proj: fill(h * (cfg.num_attention_heads * cfg.head_dim), 0.1),
            q_norm: ones(cfg.head_dim),
            k_norm: ones(cfg.head_dim),
        },
        mlp: MtpMoeWeights {
            router_gate: fill(num_experts * h, 0.1),
            experts_gate_up_proj: fill(num_experts * 2 * moe_inter * h, 0.05),
            experts_down_proj: fill(num_experts * h * moe_inter, 0.05),
            shared_gate_proj: fill(shared_inter * h, 0.05),
            shared_up_proj: fill(shared_inter * h, 0.05),
            shared_down_proj: fill(h * shared_inter, 0.05),
            shared_expert_gate: fill(h, 0.1),
        },
    };

    MtpWeights {
        fc_weight: fill(h * (2 * h), 0.1),
        layers: vec![layer],
        norm_weight: ones(h),
        pre_fc_norm_embedding_weight: ones(h),
        pre_fc_norm_hidden_weight: ones(h),
    }
}

/// A [`crate::weights::TensorSource`] that records every name passed to
/// `get_f32_tensor_owned`, so a test can assert a specific tensor's shape mismatch
/// is caught from the header-declared shape without that tensor's data ever being
/// copied (other, correctly-shaped tensors ahead of it in iteration order are still
/// fetched normally).
struct CountingTensorSource {
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    materialized: std::cell::RefCell<std::collections::HashSet<String>>,
}

impl crate::weights::TensorSource for CountingTensorSource {
    fn has_tensor(&mut self, name: &str) -> Result<bool, crate::error::InferenceError> {
        Ok(self.tensors.contains_key(name))
    }
    fn tensor_shape(
        &mut self,
        name: &str,
    ) -> Result<Option<Vec<usize>>, crate::error::InferenceError> {
        Ok(self.tensors.get(name).map(|(_, s)| s.clone()))
    }
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), crate::error::InferenceError> {
        self.materialized.borrow_mut().insert(name.to_string());
        self.tensors
            .get(name)
            .map(|(d, s)| (d.clone(), s.clone()))
            .ok_or_else(|| crate::error::InferenceError::MissingTensor(name.to_string()))
    }
}

/// Full, cfg-consistent tensor set for `tiny_mtp_config()`'s single-layer geometry,
/// keyed exactly as `MtpWeights::load_from_source` expects.
fn tiny_mtp_tensor_map(
    cfg: &MtpConfig,
) -> std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> {
    let h = cfg.hidden_size;
    let q_proj_rows = 2 * cfg.num_attention_heads * cfg.head_dim;
    let kv_proj_rows = cfg.num_key_value_heads * cfg.head_dim;
    let o_proj_cols = cfg.num_attention_heads * cfg.head_dim;
    let moe_inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;
    let num_experts = cfg.num_experts;
    let z = |n: usize| vec![0.0f32; n];

    let mut m = std::collections::HashMap::new();
    m.insert("mtp.fc.weight".to_string(), (z(h * 2 * h), vec![h, 2 * h]));
    m.insert("mtp.norm.weight".to_string(), (z(h), vec![h]));
    m.insert(
        "mtp.pre_fc_norm_embedding.weight".to_string(),
        (z(h), vec![h]),
    );
    m.insert("mtp.pre_fc_norm_hidden.weight".to_string(), (z(h), vec![h]));
    for i in 0..cfg.num_hidden_layers {
        let p = format!("mtp.layers.{i}");
        m.insert(format!("{p}.input_layernorm.weight"), (z(h), vec![h]));
        m.insert(
            format!("{p}.post_attention_layernorm.weight"),
            (z(h), vec![h]),
        );
        m.insert(
            format!("{p}.self_attn.q_proj.weight"),
            (z(q_proj_rows * h), vec![q_proj_rows, h]),
        );
        m.insert(
            format!("{p}.self_attn.k_proj.weight"),
            (z(kv_proj_rows * h), vec![kv_proj_rows, h]),
        );
        m.insert(
            format!("{p}.self_attn.v_proj.weight"),
            (z(kv_proj_rows * h), vec![kv_proj_rows, h]),
        );
        m.insert(
            format!("{p}.self_attn.o_proj.weight"),
            (z(h * o_proj_cols), vec![h, o_proj_cols]),
        );
        m.insert(
            format!("{p}.self_attn.q_norm.weight"),
            (z(cfg.head_dim), vec![cfg.head_dim]),
        );
        m.insert(
            format!("{p}.self_attn.k_norm.weight"),
            (z(cfg.head_dim), vec![cfg.head_dim]),
        );
        m.insert(
            format!("{p}.mlp.gate.weight"),
            (z(num_experts * h), vec![num_experts, h]),
        );
        m.insert(
            format!("{p}.mlp.experts.gate_up_proj"),
            (
                z(num_experts * 2 * moe_inter * h),
                vec![num_experts, 2 * moe_inter, h],
            ),
        );
        m.insert(
            format!("{p}.mlp.experts.down_proj"),
            (
                z(num_experts * h * moe_inter),
                vec![num_experts, h, moe_inter],
            ),
        );
        m.insert(
            format!("{p}.mlp.shared_expert.gate_proj.weight"),
            (z(shared_inter * h), vec![shared_inter, h]),
        );
        m.insert(
            format!("{p}.mlp.shared_expert.up_proj.weight"),
            (z(shared_inter * h), vec![shared_inter, h]),
        );
        m.insert(
            format!("{p}.mlp.shared_expert.down_proj.weight"),
            (z(h * shared_inter), vec![h, shared_inter]),
        );
        m.insert(
            format!("{p}.mlp.shared_expert_gate.weight"),
            (z(h), vec![1, h]),
        );
    }
    m
}

/// An undersized (but declared-shape-visible) `q_proj` tensor must be rejected from
/// its header-declared shape before `get_f32_tensor_owned` ever copies its data.
/// Before the fix, `load_from_source`'s `load_checked` closure materialized the
/// tensor first and only then compared shapes; that ordering still returns `Err`
/// here (so `expect_err` alone would not catch a regression) -- the "mutated name
/// was never materialized" assertion is what pins the preflight-before-load
/// contract specifically.
#[test]
fn mtp_load_from_source_rejects_undersized_tensor_before_materialization() {
    let cfg = tiny_mtp_config();
    let mut tensors = tiny_mtp_tensor_map(&cfg);
    let mutated_name = "mtp.layers.0.self_attn.q_proj.weight".to_string();
    let h = cfg.hidden_size;
    let q_proj_rows = 2 * cfg.num_attention_heads * cfg.head_dim;
    // Declared shape is undersized relative to [q_proj_rows, hidden], with data
    // matching the (wrong) declared shape so a naive `data.len()`-only check would
    // not catch it either -- only the axis-shape comparison does.
    tensors.insert(
        mutated_name.clone(),
        (
            vec![0.0f32; (q_proj_rows - 1) * h],
            vec![q_proj_rows - 1, h],
        ),
    );
    let mut source = CountingTensorSource {
        tensors,
        materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
    };

    let Err(err) = MtpWeights::load_from_source(&mut source, &cfg) else {
        panic!("an undersized required tensor must fail closed, not silently load");
    };
    assert!(
        err.to_string().contains(&mutated_name),
        "error must name the tensor: {err}"
    );
    assert!(
        !source.materialized.borrow().contains(&mutated_name),
        "the mismatched tensor's data must never be copied"
    );
}

fn dense_mtp_config() -> MtpConfig {
    MtpConfig {
        moe_intermediate_size: 3,
        shared_expert_intermediate_size: 0,
        ..tiny_mtp_config()
    }
}

/// The tensor set of a dense-MLP checkpoint: `tiny_mtp_tensor_map` with the routed MLP
/// tensors replaced by `mlp.{gate,up,down}_proj.weight`.
fn dense_mtp_tensor_map(
    cfg: &MtpConfig,
) -> std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> {
    let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let mut m = tiny_mtp_tensor_map(cfg);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("mtp.layers.{i}");
        for routed in [
            "mlp.gate.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        ] {
            m.remove(&format!("{p}.{routed}"));
        }
        m.insert(
            format!("{p}.mlp.gate_proj.weight"),
            (
                (0..inter * h).map(|v| 1.0 + v as f32).collect(),
                vec![inter, h],
            ),
        );
        m.insert(
            format!("{p}.mlp.up_proj.weight"),
            (
                (0..inter * h).map(|v| -1000.0 - v as f32).collect(),
                vec![inter, h],
            ),
        );
        m.insert(
            format!("{p}.mlp.down_proj.weight"),
            (
                (0..h * inter).map(|v| 0.5 + v as f32).collect(),
                vec![h, inter],
            ),
        );
    }
    m
}

fn counting_source(
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> CountingTensorSource {
    CountingTensorSource {
        tensors,
        materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
    }
}

fn load_err(
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    cfg: &MtpConfig,
) -> String {
    match MtpWeights::load_from_source(&mut counting_source(tensors), cfg) {
        Ok(_) => panic!("the load must fail"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn mtp_dense_layout_loads_into_one_expert_with_gate_rows_first() {
    let cfg = dense_mtp_config();
    let tensors = dense_mtp_tensor_map(&cfg);
    let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let gate = tensors["mtp.layers.0.mlp.gate_proj.weight"].0.clone();
    let up = tensors["mtp.layers.0.mlp.up_proj.weight"].0.clone();
    let down = tensors["mtp.layers.0.mlp.down_proj.weight"].0.clone();

    let w = MtpWeights::load_from_source(&mut counting_source(tensors), &cfg).unwrap();
    assert_eq!(w.layers.len(), 1);
    let mlp = &w.layers[0].mlp;
    let mut want_gate_up = gate;
    want_gate_up.extend_from_slice(&up);
    assert_eq!(mlp.experts_gate_up_proj.len(), 2 * inter * h);
    assert_eq!(mlp.experts_gate_up_proj, want_gate_up);
    assert_eq!(mlp.experts_down_proj, down);
    assert_eq!(mlp.router_gate, vec![0.0; h]);
    assert!(mlp.shared_gate_proj.is_empty());
    assert!(mlp.shared_up_proj.is_empty());
    assert!(mlp.shared_down_proj.is_empty());
    assert_eq!(mlp.shared_expert_gate.len(), h);
}

#[test]
fn mtp_routed_layout_still_loads() {
    for num_experts in [1, 2] {
        let cfg = MtpConfig {
            num_experts,
            ..tiny_mtp_config()
        };
        let mut tensors = tiny_mtp_tensor_map(&cfg);
        let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let marked: Vec<f32> = (0..num_experts * 2 * inter * h).map(|v| v as f32).collect();
        tensors.insert(
            "mtp.layers.0.mlp.experts.gate_up_proj".to_string(),
            (marked.clone(), vec![num_experts, 2 * inter, h]),
        );
        let w = MtpWeights::load_from_source(&mut counting_source(tensors), &cfg).unwrap();
        let mlp = &w.layers[0].mlp;
        assert_eq!(mlp.experts_gate_up_proj, marked, "{num_experts} experts");
        assert_eq!(mlp.router_gate.len(), num_experts * h);
        assert_eq!(
            mlp.shared_gate_proj.len(),
            cfg.shared_expert_intermediate_size * h
        );
        assert_eq!(mlp.shared_expert_gate.len(), h);
    }
}

#[test]
fn mtp_mixed_dense_and_routed_layout_errors_naming_the_layer_and_keys() {
    let cfg = tiny_mtp_config();
    let mut tensors = tiny_mtp_tensor_map(&cfg);
    let (h, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
    tensors.insert(
        "mtp.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![0.0; inter * h], vec![inter, h]),
    );
    let err = load_err(tensors, &cfg);
    assert!(err.contains("mtp.layers.0"), "{err}");
    assert!(err.contains("ambiguous"), "{err}");
    assert!(err.contains("mlp.gate_proj.weight"), "{err}");
    assert!(err.contains("mlp.experts.gate_up_proj"), "{err}");
}

#[test]
fn mtp_layer_with_neither_mlp_layout_errors_naming_the_layer() {
    let cfg = tiny_mtp_config();
    let mut tensors = tiny_mtp_tensor_map(&cfg);
    tensors.remove("mtp.layers.0.mlp.experts.gate_up_proj");
    tensors.remove("mtp.layers.0.mlp.experts.down_proj");
    let err = load_err(tensors, &cfg);
    assert!(err.contains("mtp.layers.0"), "{err}");
    assert!(err.contains("no MLP tensors found"), "{err}");
}

#[test]
fn mtp_dense_layout_rejects_a_config_that_is_not_one_expert_without_shared() {
    let dense = dense_mtp_config();
    for (what, cfg) in [
        (
            "num_experts",
            MtpConfig {
                num_experts: 2,
                ..dense.clone()
            },
        ),
        (
            "num_experts_per_tok",
            MtpConfig {
                num_experts_per_tok: 2,
                ..dense.clone()
            },
        ),
        (
            "shared_expert_intermediate_size",
            MtpConfig {
                shared_expert_intermediate_size: 2,
                ..dense.clone()
            },
        ),
    ] {
        let err = load_err(dense_mtp_tensor_map(&dense), &cfg);
        assert!(err.contains("mtp.layers.0"), "{what}: {err}");
        assert!(err.contains("dense MLP"), "{what}: {err}");
    }

    let wrong_inter = MtpConfig {
        moe_intermediate_size: dense.moe_intermediate_size + 1,
        ..dense.clone()
    };
    let err = load_err(dense_mtp_tensor_map(&dense), &wrong_inter);
    assert!(err.contains("mtp.layers.0.mlp.gate_proj.weight"), "{err}");

    let mut partial = dense_mtp_tensor_map(&dense);
    partial.remove("mtp.layers.0.mlp.up_proj.weight");
    let err = load_err(partial, &dense);
    assert!(err.contains("mtp.layers.0.mlp.up_proj.weight"), "{err}");
}

#[test]
fn mtp_config_from_qwen35_maps_dense_and_routed_models() {
    use crate::model::qwen35_config::Qwen35Config;

    let dense_model = Qwen35Config::qwen35_0_8b();
    let c = MtpConfig::from_qwen35(&dense_model, 3).unwrap();
    assert_eq!(c.draft_length, 3);
    assert_eq!(c.num_hidden_layers, dense_model.mtp_num_hidden_layers);
    assert_eq!(c.hidden_size, dense_model.hidden_size);
    assert_eq!(c.vocab_size, dense_model.vocab_size);
    assert_eq!(c.num_attention_heads, dense_model.num_attention_heads);
    assert_eq!(c.num_key_value_heads, dense_model.num_key_value_heads);
    assert_eq!(c.head_dim, dense_model.head_dim);
    assert_eq!(c.rms_norm_eps, dense_model.rms_norm_eps);
    assert_eq!(c.rope_theta, dense_model.rope_theta);
    assert_eq!(c.partial_rotary_factor, dense_model.partial_rotary_factor);
    assert_eq!(
        (
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            c.shared_expert_intermediate_size
        ),
        (1, 1, dense_model.intermediate_size, 0)
    );
    assert!(!c.use_dedicated_embeddings);

    let routed_model = Qwen35Config::qwen36_35b_a3b();
    let c = MtpConfig::from_qwen35(&routed_model, 2).unwrap();
    assert_eq!(
        (
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            c.shared_expert_intermediate_size
        ),
        (256, 8, 512, 512)
    );

    let no_mtp = Qwen35Config::qwen35_2b();
    assert_eq!(no_mtp.mtp_num_hidden_layers, 0);
    let err = MtpConfig::from_qwen35(&no_mtp, 3).unwrap_err().to_string();
    assert!(err.contains("no MTP layer"), "{err}");

    let mut half_moe = routed_model.clone();
    half_moe.num_experts_per_tok = None;
    let err = MtpConfig::from_qwen35(&half_moe, 3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("num_experts_per_tok"), "{err}");
}

/// A dense head loaded through `MtpWeights::load_from_source` must produce the logits, the
/// output hidden state and the K/V rows of a plain SwiGLU MLP computed by hand.
///
/// The reference below evaluates `down(silu(gate x) * (up x))` from the checkpoint's
/// gate/up/down tensors directly, in f64, so it does not go through the packed
/// `experts_gate_up_proj` layout, the one-expert router or the zero-width shared expert.
/// The second step's K/V row is built from the first step's output hidden state, so it
/// also depends on the MLP of step one. Errors are measured against the larger of the
/// element and the vector's largest magnitude, so a logit that cancels to near zero does
/// not turn f32 rounding into a large relative error.
#[test]
fn mtp_dense_head_matches_a_direct_swiglu_reference() {
    let cfg = dense_mtp_config();
    let (h, hd, nh, nkv, inter) = (
        cfg.hidden_size,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.moe_intermediate_size,
    );
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);

    let gen_w = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 1.7 + salt as f32 * 2.3 + 0.4).sin())
            .collect()
    };
    let gen_norm = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| -0.6 + 0.3 * (i as f32 * 2.1 + salt as f32 * 1.9 + 0.7).sin())
            .collect()
    };
    let mut tensors = std::collections::HashMap::new();
    let mut put = |name: &str, data: Vec<f32>, shape: &[usize]| {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        tensors.insert(name.to_string(), (data, shape.to_vec()));
    };
    put("mtp.fc.weight", gen_w(h * 2 * h, 1), &[h, 2 * h]);
    put("mtp.norm.weight", gen_norm(h, 17), &[h]);
    put("mtp.pre_fc_norm_embedding.weight", gen_norm(h, 18), &[h]);
    put("mtp.pre_fc_norm_hidden.weight", gen_norm(h, 19), &[h]);
    let p = "mtp.layers.0";
    put(&format!("{p}.input_layernorm.weight"), gen_norm(h, 2), &[h]);
    put(
        &format!("{p}.post_attention_layernorm.weight"),
        gen_norm(h, 3),
        &[h],
    );
    put(
        &format!("{p}.self_attn.q_proj.weight"),
        gen_w(2 * q_dim * h, 4),
        &[2 * q_dim, h],
    );
    put(
        &format!("{p}.self_attn.k_proj.weight"),
        gen_w(kv_dim * h, 5),
        &[kv_dim, h],
    );
    put(
        &format!("{p}.self_attn.v_proj.weight"),
        gen_w(kv_dim * h, 6),
        &[kv_dim, h],
    );
    put(
        &format!("{p}.self_attn.o_proj.weight"),
        gen_w(h * q_dim, 7),
        &[h, q_dim],
    );
    put(
        &format!("{p}.self_attn.q_norm.weight"),
        gen_norm(hd, 8),
        &[hd],
    );
    put(
        &format!("{p}.self_attn.k_norm.weight"),
        gen_norm(hd, 9),
        &[hd],
    );
    put(
        &format!("{p}.mlp.gate_proj.weight"),
        gen_w(inter * h, 10),
        &[inter, h],
    );
    put(
        &format!("{p}.mlp.up_proj.weight"),
        gen_w(inter * h, 11),
        &[inter, h],
    );
    put(
        &format!("{p}.mlp.down_proj.weight"),
        gen_w(h * inter, 12),
        &[h, inter],
    );
    let embed = gen_w(cfg.vocab_size * h, 20);
    let lm_head = gen_w(cfg.vocab_size * h, 21);
    let raw = |name: &str| -> Vec<f32> { tensors[name].0.clone() };

    let weights =
        MtpWeights::load_from_source(&mut counting_source(tensors.clone()), &cfg).unwrap();

    // --- Test-local reference, f64, written from the dense SwiGLU definition. ---
    let eps = cfg.rms_norm_eps as f64;
    let norm = |x: &[f64], gamma: &[f32]| -> Vec<f64> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter()
            .zip(gamma)
            .map(|(v, g)| v * inv * (1.0 + *g as f64))
            .collect()
    };
    let matvec = |w: &[f32], rows: usize, cols: usize, x: &[f64]| -> Vec<f64> {
        assert_eq!((w.len(), x.len()), (rows * cols, cols));
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * x[c]).sum())
            .collect()
    };
    let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
    let silu = |v: f64| v * sigmoid(v);
    let rot = (hd as f32 * cfg.partial_rotary_factor) as usize;
    let rope = |x: &mut [f64], pos: usize| {
        let half = rot / 2;
        let orig = x[..rot].to_vec();
        for i in 0..half {
            let angle = pos as f64 * cfg.rope_theta.powf(-(2.0 * i as f64) / rot as f64);
            let (s, c) = angle.sin_cos();
            x[i] = orig[i] * c - orig[half + i] * s;
            x[half + i] = orig[half + i] * c + orig[i] * s;
        }
    };
    let f16r = |v: f64| half::f16::from_f32(v as f32).to_f32() as f64;
    let (fc, norm_w) = (raw("mtp.fc.weight"), raw("mtp.norm.weight"));
    let (pre_emb, pre_hid) = (
        raw("mtp.pre_fc_norm_embedding.weight"),
        raw("mtp.pre_fc_norm_hidden.weight"),
    );
    let (iln, paln) = (
        raw(&format!("{p}.input_layernorm.weight")),
        raw(&format!("{p}.post_attention_layernorm.weight")),
    );
    let (wq, wk, wv, wo) = (
        raw(&format!("{p}.self_attn.q_proj.weight")),
        raw(&format!("{p}.self_attn.k_proj.weight")),
        raw(&format!("{p}.self_attn.v_proj.weight")),
        raw(&format!("{p}.self_attn.o_proj.weight")),
    );
    let (qn, kn) = (
        raw(&format!("{p}.self_attn.q_norm.weight")),
        raw(&format!("{p}.self_attn.k_norm.weight")),
    );
    let (w_gate, w_up, w_down) = (
        raw(&format!("{p}.mlp.gate_proj.weight")),
        raw(&format!("{p}.mlp.up_proj.weight")),
        raw(&format!("{p}.mlp.down_proj.weight")),
    );
    let mut kv_rows: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut ref_step = |token: usize, pos: usize, prev: &[f32]| -> (Vec<f64>, Vec<f64>) {
        let emb: Vec<f64> = embed[token * h..(token + 1) * h]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let prev: Vec<f64> = prev.iter().map(|&v| v as f64).collect();
        let cat = [norm(&emb, &pre_emb), norm(&prev, &pre_hid)].concat();
        let x = matvec(&fc, h, 2 * h, &cat);

        let a = norm(&x, &iln);
        let qg = matvec(&wq, 2 * q_dim, h, &a);
        let mut q = Vec::new();
        let mut gate = Vec::new();
        for head in 0..nh {
            q.extend_from_slice(&qg[head * 2 * hd..head * 2 * hd + hd]);
            gate.extend_from_slice(&qg[head * 2 * hd + hd..(head + 1) * 2 * hd]);
        }
        let mut k = matvec(&wk, kv_dim, h, &a);
        let v = matvec(&wv, kv_dim, h, &a);
        for head in 0..nh {
            let n = norm(&q[head * hd..(head + 1) * hd], &qn);
            q[head * hd..(head + 1) * hd].copy_from_slice(&n);
            rope(&mut q[head * hd..(head + 1) * hd], pos);
        }
        for head in 0..nkv {
            let n = norm(&k[head * hd..(head + 1) * hd], &kn);
            k[head * hd..(head + 1) * hd].copy_from_slice(&n);
            rope(&mut k[head * hd..(head + 1) * hd], pos);
        }
        kv_rows.push((
            k.iter().map(|&t| f16r(t)).collect(),
            v.iter().map(|&t| f16r(t)).collect(),
        ));

        let mut ctx = vec![0.0f64; q_dim];
        for head in 0..nh {
            let kvh = head / (nh / nkv);
            let scores: Vec<f64> = kv_rows
                .iter()
                .map(|(kr, _)| {
                    (0..hd)
                        .map(|d| q[head * hd + d] * kr[kvh * hd + d])
                        .sum::<f64>()
                        / (hd as f64).sqrt()
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..hd {
                let o: f64 = kv_rows
                    .iter()
                    .zip(&e)
                    .map(|((_, vr), w)| w / z * vr[kvh * hd + d])
                    .sum();
                ctx[head * hd + d] = o * sigmoid(gate[head * hd + d]);
            }
        }
        let attn = matvec(&wo, h, q_dim, &ctx);
        let x1: Vec<f64> = x.iter().zip(&attn).map(|(r, a)| r + a).collect();

        let m = norm(&x1, &paln);
        let g = matvec(&w_gate, inter, h, &m);
        let u = matvec(&w_up, inter, h, &m);
        let act: Vec<f64> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
        let down = matvec(&w_down, h, inter, &act);
        let x2: Vec<f64> = (0..h).map(|i| x1[i] + down[i]).collect();

        let hidden = norm(&x2, &norm_w);
        let logits = matvec(&lm_head, cfg.vocab_size, h, &hidden);
        (logits, hidden)
    };

    let worst_ratio = |got: &[f32], want: &[f64]| -> f64 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
        got.iter()
            .zip(want)
            .map(|(g, w)| (*g as f64 - w).abs() / (1e-5 * w.abs().max(scale)))
            .fold(0.0, f64::max)
    };

    let mut verifier = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, 8).unwrap();
    let (tok1, tok2) = (1u32, 2u32);
    let prev1 = [0.7f32, -1.2, 0.4, 1.6];
    let out1 = verifier.forward_one(tok1, 0, &prev1).unwrap();
    let (ref_logits1, ref_hidden1) = ref_step(tok1 as usize, 0, &prev1);
    let ref_prev2: Vec<f32> = ref_hidden1.iter().map(|&v| v as f32).collect();
    let out2 = verifier.forward_one(tok2, 1, &out1.hidden).unwrap();
    let (ref_logits2, ref_hidden2) = ref_step(tok2 as usize, 1, &ref_prev2);

    let mut worst = [
        ("step 1 logits", worst_ratio(&out1.logits, &ref_logits1)),
        ("step 1 hidden", worst_ratio(&out1.hidden, &ref_hidden1)),
        ("step 2 logits", worst_ratio(&out2.logits, &ref_logits2)),
        ("step 2 hidden", worst_ratio(&out2.hidden, &ref_hidden2)),
    ]
    .to_vec();
    for (row, (ref_k, ref_v)) in kv_rows.iter().enumerate() {
        let k: Vec<f32> = verifier.cache.k_buffer(0)[row * kv_dim..(row + 1) * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let v: Vec<f32> = verifier.cache.v_buffer(0)[row * kv_dim..(row + 1) * kv_dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        worst.push((
            if row == 0 { "K row 0" } else { "K row 1" },
            worst_ratio(&k, ref_k),
        ));
        worst.push((
            if row == 0 { "V row 0" } else { "V row 1" },
            worst_ratio(&v, ref_v),
        ));
    }
    println!("dense-head worst error / 1e-5 tolerance: {worst:?}");
    for (what, ratio) in &worst {
        assert!(
            *ratio <= 1.0,
            "{what}: the dense head differs from the direct SwiGLU reference by {ratio} \
                 times the 1e-5 relative tolerance"
        );
    }
}

#[test]
fn mtp_verifier_f16_cache_replay_after_rollback_matches_fresh_suffix() {
    let cfg = tiny_mtp_config();
    let weights = tiny_mtp_weights(&cfg);

    // Embed and lm_head: [vocab * hidden] — small non-zero values.
    let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();

    let h = cfg.hidden_size;
    let max_seq = 8usize;

    // Previous-hidden stub: same for all steps (non-zero, unit-like).
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    // --- Replay path ---
    let mut replay = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
    replay.forward_one(1, 0, &prev_hidden).unwrap();
    replay.forward_one(2, 1, &prev_hidden).unwrap();
    replay.forward_one(3, 2, &prev_hidden).unwrap();
    // Roll back to after position 0 (seq_len=1).
    replay.rollback_cache_to(1).unwrap();
    let replay_out = replay.forward_one(7, 1, &prev_hidden).unwrap();

    // --- Fresh path ---
    let mut fresh = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
    fresh.forward_one(1, 0, &prev_hidden).unwrap();
    let fresh_out = fresh.forward_one(7, 1, &prev_hidden).unwrap();

    // Both caches should now hold exactly 2 tokens.
    assert_eq!(
        replay.cache.seq_len(),
        2,
        "replay cache must hold 2 tokens after rollback+replay"
    );
    assert_eq!(fresh.cache.seq_len(), 2, "fresh cache must hold 2 tokens");

    // f16 KV slices at positions 0 and 1 must be identical.
    assert_eq!(
        replay.cache.get_k_f16(0),
        fresh.cache.get_k_f16(0),
        "K cache mismatch after rollback: stale f16 data contaminating position 0 or 1"
    );
    assert_eq!(
        replay.cache.get_v_f16(0),
        fresh.cache.get_v_f16(0),
        "V cache mismatch after rollback: stale f16 data contaminating position 0 or 1"
    );

    // Logits from the final forward_one must be identical.
    assert_eq!(
        replay_out.logits.len(),
        fresh_out.logits.len(),
        "logit vector length must match"
    );
    for (i, (r, f)) in replay_out
        .logits
        .iter()
        .zip(fresh_out.logits.iter())
        .enumerate()
    {
        assert_eq!(
            r, f,
            "logits[{i}] differ after rollback (replay={r}, fresh={f}): \
                 stale f16 dequant scratch is contaminating attention output"
        );
    }
}

/// Qwen3.5 checkpoints store every RMSNorm weight for the shifted convention
/// `x / rms(x) * (1 + gamma)`. This test compares two `forward_one` steps against a
/// test-local f64 reference written from that convention, on a fixture whose norm
/// weights centre negative and differ per site, so applying plain `gamma` at any one
/// of the seven norm sites moves the logits far outside the tolerance:
/// pre-fc embedding, pre-fc hidden, input layernorm, per-head q, per-head k,
/// post-attention layernorm and the final MTP norm.
///
/// Step 1 attends to a single key, where the softmax is 1 whatever q and k are, so it
/// cannot see the q/k norms. Step 2 attends to two keys, so both q and k norms move
/// its scores.
#[test]
fn mtp_forward_one_applies_shifted_norm_convention_at_every_site() {
    let cfg = tiny_mtp_config();
    let (h, hd, nh, nkv) = (
        cfg.hidden_size,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
    );
    let (q_dim, kv_dim) = (nh * hd, nkv * hd);
    let (moe_inter, shared_inter) = (
        cfg.moe_intermediate_size,
        cfg.shared_expert_intermediate_size,
    );
    assert_eq!((cfg.num_experts, cfg.num_experts_per_tok), (1, 1));

    let gen_w = |n: usize, salt: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| scale * (i as f32 * 1.7 + salt as f32 * 2.3 + 0.4).sin())
            .collect()
    };
    // Centred at -0.6 and non-uniform: `1 + gamma` is in [0.1, 0.7], plain `gamma` in
    // [-0.9, -0.3], so the two conventions differ in sign and in shape per channel.
    let gen_norm = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| -0.6 + 0.3 * (i as f32 * 2.1 + salt as f32 * 1.9 + 0.7).sin())
            .collect()
    };

    let weights = MtpWeights {
        fc_weight: gen_w(h * 2 * h, 1, 1.0),
        layers: vec![MtpLayerWeights {
            input_layernorm: gen_norm(h, 2),
            post_attention_layernorm: gen_norm(h, 3),
            self_attn: MtpAttentionWeights {
                q_proj: gen_w(2 * q_dim * h, 4, 1.0),
                k_proj: gen_w(kv_dim * h, 5, 1.0),
                v_proj: gen_w(kv_dim * h, 6, 1.0),
                o_proj: gen_w(h * q_dim, 7, 1.0),
                q_norm: gen_norm(hd, 8),
                k_norm: gen_norm(hd, 9),
            },
            mlp: MtpMoeWeights {
                router_gate: gen_w(h, 10, 1.0),
                experts_gate_up_proj: gen_w(2 * moe_inter * h, 11, 1.0),
                experts_down_proj: gen_w(h * moe_inter, 12, 1.0),
                shared_gate_proj: gen_w(shared_inter * h, 13, 1.0),
                shared_up_proj: gen_w(shared_inter * h, 14, 1.0),
                shared_down_proj: gen_w(h * shared_inter, 15, 1.0),
                shared_expert_gate: gen_w(h, 16, 1.0),
            },
        }],
        norm_weight: gen_norm(h, 17),
        pre_fc_norm_embedding_weight: gen_norm(h, 18),
        pre_fc_norm_hidden_weight: gen_norm(h, 19),
    };
    let embed = gen_w(cfg.vocab_size * h, 20, 1.0);
    let lm_head = gen_w(cfg.vocab_size * h, 21, 1.0);

    // --- Test-local reference, f64, written from the shifted convention. ---
    let eps = cfg.rms_norm_eps as f64;
    let norm = |x: &[f64], gamma: &[f32]| -> Vec<f64> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter()
            .zip(gamma)
            .map(|(v, g)| v * inv * (1.0 + *g as f64))
            .collect()
    };
    // y = W x, W row-major [rows, cols]
    let matvec = |w: &[f32], rows: usize, cols: usize, x: &[f64]| -> Vec<f64> {
        assert_eq!((w.len(), x.len()), (rows * cols, cols));
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * x[c]).sum())
            .collect()
    };
    let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
    let silu = |v: f64| v * sigmoid(v);
    let rot = (hd as f32 * cfg.partial_rotary_factor) as usize;
    // x * cos + rotate_half(x) * sin on the first `rot` dims, rotate_half(x) = [-x2, x1].
    let rope = |x: &mut [f64], pos: usize| {
        let half = rot / 2;
        let orig = x[..rot].to_vec();
        for i in 0..half {
            let angle = pos as f64 * cfg.rope_theta.powf(-(2.0 * i as f64) / rot as f64);
            let (s, c) = angle.sin_cos();
            x[i] = orig[i] * c - orig[half + i] * s;
            x[half + i] = orig[half + i] * c + orig[i] * s;
        }
    };
    // The verifier stores K/V in an f16 cache; round the same way.
    let f16r = |v: f64| half::f16::from_f32(v as f32).to_f32() as f64;
    let mut kv_rows: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut ref_step = |token: usize, pos: usize, prev: &[f32]| -> (Vec<f64>, Vec<f64>) {
        let l = &weights.layers[0];
        let emb: Vec<f64> = embed[token * h..(token + 1) * h]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let prev: Vec<f64> = prev.iter().map(|&v| v as f64).collect();
        let cat = [
            norm(&emb, &weights.pre_fc_norm_embedding_weight),
            norm(&prev, &weights.pre_fc_norm_hidden_weight),
        ]
        .concat();
        let x = matvec(&weights.fc_weight, h, 2 * h, &cat);

        let a = norm(&x, &l.input_layernorm);
        let qg = matvec(&l.self_attn.q_proj, 2 * q_dim, h, &a);
        let mut q = Vec::new();
        let mut gate = Vec::new();
        for head in 0..nh {
            q.extend_from_slice(&qg[head * 2 * hd..head * 2 * hd + hd]);
            gate.extend_from_slice(&qg[head * 2 * hd + hd..(head + 1) * 2 * hd]);
        }
        let mut k = matvec(&l.self_attn.k_proj, kv_dim, h, &a);
        let v = matvec(&l.self_attn.v_proj, kv_dim, h, &a);
        for head in 0..nh {
            let n = norm(&q[head * hd..(head + 1) * hd], &l.self_attn.q_norm);
            q[head * hd..(head + 1) * hd].copy_from_slice(&n);
            rope(&mut q[head * hd..(head + 1) * hd], pos);
        }
        for head in 0..nkv {
            let n = norm(&k[head * hd..(head + 1) * hd], &l.self_attn.k_norm);
            k[head * hd..(head + 1) * hd].copy_from_slice(&n);
            rope(&mut k[head * hd..(head + 1) * hd], pos);
        }
        kv_rows.push((
            k.iter().map(|&t| f16r(t)).collect(),
            v.iter().map(|&t| f16r(t)).collect(),
        ));

        let mut ctx = vec![0.0f64; q_dim];
        for head in 0..nh {
            let kvh = head / (nh / nkv);
            let scores: Vec<f64> = kv_rows
                .iter()
                .map(|(kr, _)| {
                    (0..hd)
                        .map(|d| q[head * hd + d] * kr[kvh * hd + d])
                        .sum::<f64>()
                        / (hd as f64).sqrt()
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..hd {
                let o: f64 = kv_rows
                    .iter()
                    .zip(&e)
                    .map(|((_, vr), w)| w / z * vr[kvh * hd + d])
                    .sum();
                ctx[head * hd + d] = o * sigmoid(gate[head * hd + d]);
            }
        }
        let attn = matvec(&l.self_attn.o_proj, h, q_dim, &ctx);
        let x1: Vec<f64> = x.iter().zip(&attn).map(|(r, a)| r + a).collect();

        let m = norm(&x1, &l.post_attention_layernorm);
        // One expert: router softmax is 1.0 and the top-1 renormalisation keeps it there.
        let gu = matvec(&l.mlp.experts_gate_up_proj, 2 * moe_inter, h, &m);
        let act: Vec<f64> = (0..moe_inter)
            .map(|j| silu(gu[j]) * gu[moe_inter + j])
            .collect();
        let expert = matvec(&l.mlp.experts_down_proj, h, moe_inter, &act);
        let sg = matvec(&l.mlp.shared_gate_proj, shared_inter, h, &m);
        let su = matvec(&l.mlp.shared_up_proj, shared_inter, h, &m);
        let sact: Vec<f64> = (0..shared_inter).map(|j| silu(sg[j]) * su[j]).collect();
        let shared = matvec(&l.mlp.shared_down_proj, h, shared_inter, &sact);
        let gate_scalar = sigmoid(
            m.iter()
                .zip(&l.mlp.shared_expert_gate)
                .map(|(a, b)| a * *b as f64)
                .sum(),
        );
        let x2: Vec<f64> = (0..h)
            .map(|i| x1[i] + expert[i] + gate_scalar * shared[i])
            .collect();

        let hidden = norm(&x2, &weights.norm_weight);
        let logits = matvec(&lm_head, cfg.vocab_size, h, &hidden);
        (logits, hidden)
    };

    let steps: [(u32, usize, [f32; 4]); 2] = [
        (1, 0, [0.7, -1.2, 0.4, 1.6]),
        (2, 1, [-0.9, 0.5, 1.3, -0.3]),
    ];
    let mut verifier = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, 8).unwrap();
    const TOL_ABS: f64 = 1e-4;
    const TOL_REL: f64 = 1e-4;
    let mut worst = (0.0f64, String::new());
    for (step, (token, pos, prev)) in steps.iter().enumerate() {
        let out = verifier.forward_one(*token, *pos, prev).unwrap();
        let (ref_logits, ref_hidden) = ref_step(*token as usize, *pos, prev);
        for (name, got, want) in [
            ("logits", &out.logits, &ref_logits),
            ("hidden", &out.hidden, &ref_hidden),
        ] {
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                let err = (*g as f64 - w).abs();
                let ratio = err / (TOL_ABS + TOL_REL * w.abs());
                if ratio > worst.0 {
                    worst = (
                        ratio,
                        format!(
                            "step {} {name}[{i}]: verifier {g} vs shifted-convention reference {w} (|diff| {err:.3e})",
                            step + 1
                        ),
                    );
                }
            }
        }
    }
    assert!(
        worst.0 <= 1.0,
        "{}; a norm site is applying plain gamma instead of 1 + gamma",
        worst.1
    );
}

#[test]
fn mtp_verifier_forward_one_at_capacity_returns_err_not_panic() {
    let cfg = tiny_mtp_config();
    let weights = tiny_mtp_weights(&cfg);

    let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();

    let h = cfg.hidden_size;
    let max_seq = 4usize;
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let mut v = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();

    // Fill the cache to exactly capacity: each forward_one appends one token.
    for pos in 0..max_seq {
        v.forward_one((pos % 2) as u32 + 1, pos, &prev_hidden)
            .unwrap_or_else(|e| {
                panic!("forward_one at pos {pos} (below capacity) should succeed: {e:?}")
            });
    }
    assert_eq!(
        v.cache.seq_len(),
        max_seq,
        "cache must be full after filling"
    );
    assert!(v.cache.is_full(), "cache must report full at capacity");

    // The next forward_one would index the raw K/V buffer out of bounds
    // (it writes at seq_len * kv_dim before advance_by). It must fail closed
    // with a typed error, never panic — matches the no-panic library contract
    // and the #290 FlatKVCache hardening.
    match v.forward_one(1, max_seq, &prev_hidden) {
        Ok(_) => panic!("forward_one at capacity must return Err, not overwrite or panic"),
        Err(crate::InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("full"),
                "error message should explain the cache is full, got: {msg}"
            );
        }
        Err(other) => panic!("expected InvalidInput for full cache, got {other:?}"),
    }
}

/// forward_one returns Err (not a panic) when `position` equals max_seq_len even
/// though the KV cache is not yet full.  This guards the RoPE table OOB path in
/// mtp_apply_partial_rope (independent `position` param, distinct from the #290
/// is_full guard).
#[test]
fn mtp_verifier_forward_one_out_of_range_position_returns_err() {
    let cfg = tiny_mtp_config();
    let weights = tiny_mtp_weights(&cfg);

    let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();

    let h = cfg.hidden_size;
    let max_seq = 4usize;
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    // Cache starts empty (seq_len == 0); pass position == max_seq so the
    // cache is not full but the position is past the RoPE table.
    let mut v = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();
    assert_eq!(v.cache.seq_len(), 0, "cache must start empty");
    assert!(!v.cache.is_full(), "cache must not be full yet");

    match v.forward_one(1, max_seq, &prev_hidden) {
        Ok(_) => {
            panic!("forward_one with out-of-range position must return Err, not panic or succeed")
        }
        Err(crate::InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("position"),
                "error message should mention position, got: {msg}"
            );
        }
        Err(other) => panic!("expected InvalidInput for out-of-range position, got {other:?}"),
    }
}

/// MtpVerifier::new rejects a config whose partial_rotary_factor is outside [0,1].
/// Factor 2.0 would produce rope_dim = 2 * head_dim, making mtp_apply_partial_rope
/// index head_vec[head_dim] on a head_dim-length slice (Bug 2 / scalar validation).
#[test]
fn mtp_new_rejects_partial_rotary_factor_above_one() {
    let mut cfg = tiny_mtp_config();
    cfg.partial_rotary_factor = 2.0;
    let weights = tiny_mtp_weights(&cfg);
    let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
        panic!("partial_rotary_factor > 1 must be rejected");
    };
    assert!(
        format!("{err:?}").contains("partial_rotary_factor"),
        "expected partial_rotary_factor rejection, got: {err:?}"
    );
}

/// MtpVerifier::new rejects a config with num_key_value_heads == 0.
/// Zero kv heads causes divide-by-zero at `groups = num_q_heads / num_kv_heads`
/// on the first forward_one (Bug 2 / scalar validation).
#[test]
fn mtp_new_rejects_zero_kv_heads() {
    let mut cfg = tiny_mtp_config();
    cfg.num_key_value_heads = 0;
    let weights = tiny_mtp_weights(&cfg);
    let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
        panic!("num_key_value_heads == 0 must be rejected");
    };
    assert!(
        format!("{err:?}").contains("num_key_value_heads"),
        "expected num_key_value_heads rejection, got: {err:?}"
    );
}

/// MtpVerifier::new rejects zero-width RoPE (partial_rotary_factor == 0.0,
/// which the [0,1] scalar check accepts) before constructing a degenerate
/// RopeTable whose max_positions() == 0 would reject every forward_one
/// position (#362 Major: position-bound exactness for zero-width RoPE).
#[test]
fn mtp_new_rejects_zero_partial_rotary_factor() {
    let mut cfg = tiny_mtp_config();
    cfg.partial_rotary_factor = 0.0;
    let weights = tiny_mtp_weights(&cfg);
    let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
    let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
        panic!("zero-width RoPE (partial_rotary_factor == 0) must be rejected");
    };
    assert!(
        format!("{err:?}").contains("RoPE width"),
        "expected zero-width RoPE rejection, got: {err:?}"
    );
}

/// MtpVerifier::new returns Err (not a debug overflow-panic / release wrap)
/// when vocab_size * hidden_size overflows usize. Weights/embeds are tiny;
/// the checked product fires before any tensor-shape comparison
/// (#362 Medium: unchecked dimension products).
#[test]
fn mtp_new_rejects_dimension_product_overflow() {
    let base = tiny_mtp_config();
    let weights = tiny_mtp_weights(&base);
    let embed: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
    let lm_head: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
    let mut cfg = tiny_mtp_config();
    cfg.hidden_size = usize::MAX / 2 + 1;
    cfg.vocab_size = 2;
    let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
        panic!("overflowing vocab_size * hidden_size must return Err, not panic");
    };
    assert!(
        format!("{err:?}").contains("overflows usize"),
        "expected overflow rejection, got: {err:?}"
    );
}

/// MtpVerifier::new returns Err (not a panic) when num_attention_heads *
/// head_dim overflows. Reachable with TINY weights — only weights.layers.len()
/// is checked before MtpScratch/cache construction, so an oversized
/// num_attention_heads passes the scalar/divisibility checks (kv_heads=1)
/// yet overflows q_dim (#362 Medium: derived-dimension products).
#[test]
fn mtp_new_rejects_head_dim_product_overflow() {
    let base = tiny_mtp_config();
    let weights = tiny_mtp_weights(&base);
    let embed: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
    let lm_head: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
    let mut cfg = tiny_mtp_config();
    cfg.num_key_value_heads = 1;
    cfg.num_attention_heads = usize::MAX / 4 + 1; // * head_dim (4) overflows
    let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
        panic!("overflowing num_attention_heads * head_dim must return Err, not panic");
    };
    assert!(
        format!("{err:?}").contains("overflows usize"),
        "expected dimension-product overflow rejection, got: {err:?}"
    );
}

#[test]
fn generate_with_speculation_empty_prompt_returns_error() {
    let result = generate_with_speculation(
        &[],
        8,
        999,
        |_tok, _pos| panic!("forward must not run for an empty prompt"),
        3,
        4,
    );
    assert!(matches!(
        result,
        Err(crate::error::InferenceError::Inference(ref message)) if message == "empty prompt"
    ));
}

/// ADR-080 C1 (#785): `MtpVerifier::forward_one`'s own GQA
/// attention over its KV cache is a production attention row, not a
/// vocabulary/logit softmax (the PR body's original classification was wrong).
/// A NaN cached K row must zero the whole attention-context row for every
/// query head that attends through that KV head, while a sibling query head
/// attending through an unpoisoned KV head stays finite/normalized.
///
/// Uses `num_key_value_heads == num_attention_heads == 2` (MHA, `groups == 1`)
/// so each query head has its own independent KV head slice: poisoning KV
/// head 0 cannot leak into query head 1's row through GQA sharing.
#[test]
fn mtp_forward_one_nan_cached_k_row_fails_closed_sibling_head_stays_finite() {
    let mut cfg = tiny_mtp_config();
    cfg.num_key_value_heads = 2; // MHA: qh 0 -> kvh 0, qh 1 -> kvh 1, independent caches
    let weights = tiny_mtp_weights(&cfg);
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let max_seq = 8usize;

    let embed: Vec<f32> = (0..cfg.vocab_size * h)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head: Vec<f32> = (0..cfg.vocab_size * h)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();
    let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let mut verifier = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();

    // Step 0: append a real (unpoisoned) K/V pair at cache position 0.
    verifier.forward_one(1, 0, &prev_hidden).unwrap();
    assert_eq!(verifier.cache.seq_len(), 1);

    // Directly poison KV head 0's cached K vector at position 0 -- a NaN
    // score that must be caught by `row_max_and_any_nan`, not silently
    // dropped by the old `if s > max_score` fold.
    {
        let k_buf = verifier.cache.k_buffer_mut(0);
        // Position 0, KV head 0 occupies the first `head_dim` lanes of the
        // per-position `kv_dim`-wide slice.
        k_buf[0] = half::f16::NAN;
    }

    // Step 1: a second forward_one attends over cache positions {0, 1}.
    // qh=0 (kvh=0) sees the poisoned position-0 score; qh=1 (kvh=1) does not.
    verifier.forward_one(2, 1, &prev_hidden).unwrap();

    let ctx = &verifier.scratch.context;
    let poisoned_head = &ctx[0..head_dim];
    let clean_head = &ctx[head_dim..2 * head_dim];

    assert!(
        poisoned_head.iter().all(|&v| v == 0.0),
        "query head attending through the poisoned KV head must be exactly \
             all-zero (fail-closed), got {poisoned_head:?}"
    );
    assert!(
        clean_head.iter().all(|v| v.is_finite()),
        "sibling query head attending through an unpoisoned KV head must \
             stay finite, got {clean_head:?}"
    );
    assert!(
        clean_head.iter().any(|&v| v != 0.0),
        "sibling query head's context must be a real (non-degenerate) \
             normalized result, not incidentally all-zero, got {clean_head:?}"
    );
}

/// The pre-centering formula: scale first, then softmax.
fn softmax_scaled_uncentered(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv = 1.0 / temperature;
    let scaled: Vec<f32> = logits.iter().map(|&l| l * inv).collect();
    let mut out = vec![0.0f32; logits.len()];
    softmax_into(&scaled, &mut out);
    out
}

#[test]
fn softmax_into_temperature_huge_finite_logit_below_one_stays_finite() {
    let logits = [3.0e38f32, 0.0];
    let mut scratch = [0.0f32; 2];
    let mut out = [0.0f32; 2];
    softmax_into_temperature(&logits, 0.5, &mut scratch, &mut out);
    assert!(
        out.iter().all(|p| p.is_finite()),
        "scaling a huge finite logit by 1/T must not poison the row, got {out:?}"
    );
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "distribution must normalize, sum {sum}"
    );
    assert!(
        (out[0] - 1.0).abs() < 1e-6,
        "all mass on index 0, got {out:?}"
    );
    assert_eq!(out[1], 0.0);
}

#[test]
fn softmax_into_temperature_at_or_above_one_matches_uncentered_formula_bitwise() {
    let logits = [1.5f32, -2.25, 0.0, 7.0, 3.125, -0.5];
    for &t in &[1.0f32, 1.5, 2.0, 10.0] {
        let mut scratch = [0.0f32; 6];
        let mut out = [0.0f32; 6];
        softmax_into_temperature(&logits, t, &mut scratch, &mut out);
        let expected = if t == 1.0 {
            let mut e = vec![0.0f32; logits.len()];
            softmax_into(&logits, &mut e);
            e
        } else {
            softmax_scaled_uncentered(&logits, t)
        };
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "T={t} index {i}: {a} vs {b}");
        }
    }
}

#[test]
fn softmax_into_temperature_neg_inf_entry_keeps_zero_probability_below_one() {
    let logits = [2.0f32, f32::NEG_INFINITY, 1.0];
    let mut scratch = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    softmax_into_temperature(&logits, 0.5, &mut scratch, &mut out);
    assert_eq!(
        out[1], 0.0,
        "-inf entry must keep probability 0, got {out:?}"
    );
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "sum {sum}");
    assert!(out[0] > out[2]);
}
