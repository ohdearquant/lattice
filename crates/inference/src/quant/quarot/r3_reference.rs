//! R3 orientation reference — issue #703 PR1's named open question.
//!
//! The design doc (§B "Insertion-point analysis", §D "R3 mapping with GQA,
//! cache and gated attention is underspecified") flags that QuaRot's R3
//! ("Hadamard heads" on the attention output) has more than one plausible
//! insertion point once Qwen3.5's output-gated attention is in the picture:
//! rotate the cached value stream, rotate the gated context right before
//! `o_proj`, or some mix. Only one of those is an exact numerical
//! reparameterization of the dense (unrotated) forward — the others silently
//! break because a Hadamard rotation does not commute with the per-dimension
//! sigmoid gate (`ctx *= sigmoid(gate_z)`) unless the rotation is inserted
//! **after** the gate is applied.
//!
//! ## The contract this module proves and records
//!
//! For one full-attention Qwen3.5-0.8B layer's attention-axis shape (GQA:
//! `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, so
//! `q_dim=2048`, `kv_dim=512`, groups=4 — read from
//! [`crate::model::qwen35_config::Qwen35Config::qwen35_0_8b`]; the test
//! fixture's `HIDDEN` constant is intentionally smaller than the real
//! `hidden_size=1024` — see that constant's doc for why), the winning
//! orientation is:
//!
//! **R3 rotates the gated attention output (`context * sigmoid(gate_z)`),
//! immediately before `o_proj`. `o_proj` absorbs the rotation's transpose on
//! its INPUT side** — i.e. exactly [`crate::quant::quarot::plan::AbsorptionSide::InputSide`]
//! applied to `o_proj`'s `q_dim` axis, matching `rotation.rs`'s documented
//! `y = W · R^T · (R · x)` identity with `x` = the gated context. This is
//! recorded in [`crate::quant::quarot::plan::OnlineRotationSpec::r3_full_attention`]
//! as `side: AbsorptionSide::InputSide`.
//!
//! ## The rotation is cross-head, not intra-head
//!
//! The Kronecker **axis** matters and is easy to get backwards: QuaRot's
//! online "Hadamard heads" operation (paper Eq. 9, arXiv:2404.00456 Stage
//! 1c) is `Z ← Z · (H_num_heads ⊗ I_head_dim)` — it mixes values **across
//! heads at each fixed within-head channel index**, not within a single
//! head's own channels. Verbatim from the paper: "it remains to apply
//! `(H_nh ⊗ I)` to `W_out`, which results in a complete transformation of
//! `W_out ← H W_out`, and to insert a block into the forward pass that
//! computes `Z ← Z(H_nh ⊗ I)` where `Z` is the attention activation. This
//! block is denoted Hadamard heads in Figure 6." The per-head block-diagonal
//! `(I_num_heads ⊗ H_head_dim)` factor from the same section (Eq. 7-8) is a
//! *different*, purely offline operation fused into `W_v`/`W_out` with zero
//! online cost — it is not what "Hadamard heads" refers to, and a rotation
//! built from independently-seeded per-head `head_dim` blocks (the
//! `I ⊗ H_head_dim` axis) is a different transform than QuaRot's R3, even
//! though — as this module's own enumeration test demonstrates — either
//! orthogonal transform reparameterizes exactly through `o_proj`'s
//! counter-rotation and so cannot be told apart by a cancellation test
//! alone. `candidate_post_gate_context` below therefore builds the rotation
//! from a genuine cross-head [`RandomizedHadamard`] over the
//! `num_attention_heads` axis (applied identically to every `head_dim`
//! channel), not a [`BlockHadamard`] over contiguous `head_dim` blocks.
//!
//! A cache-side (value-path) rotation is *compatible* with this orientation
//! **only if it is inverted before the gate is applied** — rotating V at
//! cache-write time and then undoing that exact rotation on `context` before
//! gating is a mathematical no-op on the forward pass (see the
//! `value_round_trip_then_post_gate` candidate below), so a future PR can use
//! it to reduce KV-cache outliers without changing this contract. Rotating V
//! and leaving that rotation in place through the gate does NOT work — the
//! gate's per-dimension sigmoid does not commute with a cross-dimension
//! Hadamard mix (see the `pre_gate_value_only` candidate). The value-path
//! round trip in `value_round_trip_then_post_gate` uses a per-KV-head
//! `head_dim`-block rotation (QuaRot's *offline* Eq. 7-8 factor) purely as a
//! KV-cache-outlier companion; it is independent of, and composes cleanly
//! with, the cross-head R3 rotation applied afterward.
//!
//! ## Scope note
//!
//! RoPE is intentionally omitted from this synthetic forward: RoPE only
//! touches Q/K (the score computation), which is orthogonal to the question
//! this module answers (whether a rotation of the value/context/gate/O path
//! commutes with the gate). Q/K are still projected and used to produce a
//! non-trivial 3-position softmax, so the attention-weighted-sum step is
//! exercised for real, not stubbed to an identity.

#[cfg(test)]
mod tests {
    use crate::quant::quarot::hadamard::{BlockHadamard, RandomizedHadamard, derive_block_seed};
    use crate::quant::quarot::plan::{AbsorptionSide, OnlineRotationSpec};

    // HIDDEN is deliberately NOT the real Qwen3.5-0.8B hidden_size (1024).
    // This module's contract is about the ATTENTION-axis question (does the
    // R3 rotation mix across heads or within a head, applied before o_proj)
    // — HIDDEN only sets the width of q_gate_proj's input / o_proj's output,
    // which never participates in that question and is never itself
    // Hadamard-rotated. A small HIDDEN preserves full GQA/head-shape
    // coverage (NUM_Q_HEADS/NUM_KV_HEADS/HEAD_DIM/GROUPS below are the real
    // qwen35_0_8b values) while keeping the O(HIDDEN * Q_DIM) matvecs and
    // per-row rotation absorption — the dominant cost of this reference
    // suite (0.85s of 1.26s total QuaRot test time)
    // — cheap. Absorption is row-wise (every o_proj row gets the identical
    // treatment), so a small representative row count exercises the same
    // code path as HIDDEN=1024 without the wasted work.
    const HIDDEN: usize = 32;
    const NUM_Q_HEADS: usize = 8;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 2048
    const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512
    const GROUPS: usize = NUM_Q_HEADS / NUM_KV_HEADS; // 4
    const SEQ_LEN: usize = 3;
    const R_V_SEED: u64 = 0xA1;
    const R_O_SEED: u64 = 0xB2;

    fn synthetic_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    /// row-major [rows x cols] @ x[cols] -> y[rows]
    fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
            .collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Fixture: one full-attention layer's worth of synthetic weights + a
    /// 3-position KV cache (2 prior positions + the current token).
    struct Fixture {
        x: Vec<f32>,            // [HIDDEN] current-token input
        q_gate_proj: Vec<f32>,  // [2*Q_DIM, HIDDEN] fused Q+gate, real Qwen3.5 layout
        o_proj: Vec<f32>,       // [HIDDEN, Q_DIM]
        k_cache: Vec<Vec<f32>>, // SEQ_LEN x [KV_DIM]
        v_cache: Vec<Vec<f32>>, // SEQ_LEN x [KV_DIM]
    }

    fn build_fixture() -> Fixture {
        Fixture {
            x: synthetic_vec(HIDDEN, 1),
            q_gate_proj: synthetic_vec(2 * Q_DIM * HIDDEN, 2),
            o_proj: synthetic_vec(HIDDEN * Q_DIM, 3),
            k_cache: vec![
                synthetic_vec(KV_DIM, 10),
                synthetic_vec(KV_DIM, 11),
                synthetic_vec(KV_DIM, 12),
            ],
            v_cache: vec![
                synthetic_vec(KV_DIM, 20),
                synthetic_vec(KV_DIM, 21),
                synthetic_vec(KV_DIM, 22),
            ],
        }
    }

    /// Q + gate projection, scattered per-head exactly like
    /// `forward/cpu_f16.rs::full_attention_step_f16`: view(heads, 2*head_dim)
    /// -> chunk(2) -> [Q_h, gate_h].
    fn project_q_and_gate(fx: &Fixture) -> (Vec<f32>, Vec<f32>) {
        let q_and_gate = matvec(&fx.q_gate_proj, 2 * Q_DIM, HIDDEN, &fx.x);
        let mut q = vec![0.0f32; Q_DIM];
        let mut gate = vec![0.0f32; Q_DIM];
        for h in 0..NUM_Q_HEADS {
            let src = h * HEAD_DIM * 2;
            let dst = h * HEAD_DIM;
            q[dst..dst + HEAD_DIM].copy_from_slice(&q_and_gate[src..src + HEAD_DIM]);
            gate[dst..dst + HEAD_DIM]
                .copy_from_slice(&q_and_gate[src + HEAD_DIM..src + HEAD_DIM * 2]);
        }
        (q, gate)
    }

    /// Weighted sum of `v_cache` for every Q head (GQA: `kvh = qh / GROUPS`),
    /// scoring against the fixture's fixed `k_cache` (K/scores are unaffected
    /// by any R3 candidate — only V/context/gate/O are in question).
    fn attention_context(fx: &Fixture, q: &[f32], v_cache: &[Vec<f32>]) -> Vec<f32> {
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut context = vec![0.0f32; Q_DIM];
        for qh in 0..NUM_Q_HEADS {
            let kvh = qh / GROUPS;
            let q_off = qh * HEAD_DIM;
            let qh_vec = &q[q_off..q_off + HEAD_DIM];

            let mut scores = [0.0f32; SEQ_LEN];
            for t in 0..SEQ_LEN {
                let k_off = kvh * HEAD_DIM;
                let dot: f32 = (0..HEAD_DIM)
                    .map(|d| qh_vec[d] * fx.k_cache[t][k_off + d])
                    .sum();
                scores[t] = dot * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }

            let ctx_off = qh * HEAD_DIM;
            for d in 0..HEAD_DIM {
                let v_off = kvh * HEAD_DIM;
                let sum: f32 = (0..SEQ_LEN)
                    .map(|t| scores[t] * v_cache[t][v_off + d])
                    .sum();
                context[ctx_off + d] = sum;
            }
        }
        context
    }

    /// Dense (unrotated) reference forward: mirrors
    /// `full_attention_step_f16`'s math (minus RoPE — see module doc).
    fn dense_forward(fx: &Fixture) -> Vec<f32> {
        let (q, gate) = project_q_and_gate(fx);
        let context = attention_context(fx, &q, &fx.v_cache);
        let gated: Vec<f32> = context
            .iter()
            .zip(gate.iter())
            .map(|(&c, &g)| c * sigmoid(g))
            .collect();
        matvec(&fx.o_proj, HIDDEN, Q_DIM, &gated)
    }

    /// Apply QuaRot's online "Hadamard heads" cross-head factor
    /// `H_num_heads ⊗ I_head_dim` to a `[num_heads * head_dim]`-shaped
    /// activation in place: for each fixed within-head channel index `d`,
    /// gather the `num_heads` values at that channel across every head,
    /// rotate them with the SAME `num_heads`-dimensional
    /// [`RandomizedHadamard`], and scatter back. This mixes values ACROSS
    /// heads at a fixed channel — the opposite axis from `BlockHadamard`
    /// (which mixes channels WITHIN one head and leaves other heads alone).
    fn apply_cross_head_rotation(
        data: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        r: &RandomizedHadamard,
    ) {
        assert_eq!(data.len(), num_heads * head_dim);
        assert_eq!(r.dim(), num_heads);
        let mut channel = vec![0.0f32; num_heads];
        for d in 0..head_dim {
            for h in 0..num_heads {
                channel[h] = data[h * head_dim + d];
            }
            r.apply(&mut channel).unwrap();
            for h in 0..num_heads {
                data[h * head_dim + d] = channel[h];
            }
        }
    }

    /// Absorb the cross-head rotation (see [`apply_cross_head_rotation`])
    /// into the INPUT side of a row-major `[rows x cols]` weight
    /// (`cols = num_heads * head_dim`), mirroring
    /// `rotation::absorb_input_rotation`'s per-row identity but applying
    /// the cross-head factor to each row instead of a single dense
    /// rotation.
    fn absorb_input_cross_head_rotation(
        weight: &mut [f32],
        rows: usize,
        cols: usize,
        num_heads: usize,
        head_dim: usize,
        r: &RandomizedHadamard,
    ) {
        assert_eq!(weight.len(), rows * cols);
        assert_eq!(cols, num_heads * head_dim);
        for row in 0..rows {
            let slice = &mut weight[row * cols..(row + 1) * cols];
            apply_cross_head_rotation(slice, num_heads, head_dim, r);
        }
    }

    /// A `Q_DIM`-shaped rotation whose per-Q-head `HEAD_DIM` block is the
    /// SAME [`RandomizedHadamard`] as its KV head's `KV_DIM`-shaped block
    /// (`kvh = qh / GROUPS`), used to broadcast a value-cache rotation
    /// (defined over `KV_DIM`) onto the `Q_DIM`-shaped context/gate space so
    /// it can be inverted or absorbed there. `BlockHadamard` itself has no
    /// public "reuse another instance's per-block signs" constructor, so
    /// this test-only helper reconstructs each block directly.
    struct KvBroadcastRotation {
        per_kv_head: Vec<RandomizedHadamard>,
    }

    impl KvBroadcastRotation {
        /// `seed` must match the `KV_DIM`-shaped `BlockHadamard::new(seed,
        /// KV_DIM, HEAD_DIM)` this broadcasts. Uses the SAME
        /// `derive_block_seed` helper as `BlockHadamard` itself (daemon
        /// review minor: a duplicated derivation here would silently drift
        /// if the mixing ever changes again).
        fn matching(seed: u64) -> Self {
            let per_kv_head = (0..NUM_KV_HEADS)
                .map(|kvh| RandomizedHadamard::new(derive_block_seed(seed, kvh), HEAD_DIM).unwrap())
                .collect();
            Self { per_kv_head }
        }

        fn apply(&self, data: &mut [f32]) {
            for qh in 0..NUM_Q_HEADS {
                let kvh = qh / GROUPS;
                let slice = &mut data[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
                self.per_kv_head[kvh].apply(slice).unwrap();
            }
        }

        fn apply_inverse(&self, data: &mut [f32]) {
            for qh in 0..NUM_Q_HEADS {
                let kvh = qh / GROUPS;
                let slice = &mut data[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
                self.per_kv_head[kvh].apply_inverse(slice).unwrap();
            }
        }

        fn absorb_into_input_side(&self, weight: &mut [f32], rows: usize, cols: usize) {
            assert_eq!(weight.len(), rows * cols);
            assert_eq!(cols, Q_DIM);
            for row in 0..rows {
                let slice = &mut weight[row * cols..(row + 1) * cols];
                self.apply(slice);
            }
        }
    }

    /// Candidate 1 — rotate V pre-cache (per-KV-head block Hadamard), leave
    /// the gate untouched, counter-rotate `o_proj`'s input side with the
    /// SAME per-head rotation broadcast across Q-head groups. Expected to
    /// diverge: rotating V rotates `context` before the gate, and the gate
    /// (`ctx *= sigmoid(gate_z)`, a *different* value per dimension within a
    /// head) does not commute with a cross-dimension Hadamard mix.
    fn candidate_pre_gate_value_only(fx: &Fixture) -> Vec<f32> {
        let r_v = BlockHadamard::new(R_V_SEED, KV_DIM, HEAD_DIM).unwrap();
        let mut v_rotated = fx.v_cache.clone();
        for v in v_rotated.iter_mut() {
            r_v.apply(v).unwrap();
        }
        let (q, gate) = project_q_and_gate(fx);
        let context_rot = attention_context(fx, &q, &v_rotated);
        let gated: Vec<f32> = context_rot
            .iter()
            .zip(gate.iter())
            .map(|(&c, &g)| c * sigmoid(g))
            .collect();

        let broadcast = KvBroadcastRotation::matching(R_V_SEED);
        let mut o_rot = fx.o_proj.clone();
        broadcast.absorb_into_input_side(&mut o_rot, HIDDEN, Q_DIM);
        matvec(&o_rot, HIDDEN, Q_DIM, &gated)
    }

    /// Candidate 2 (winning orientation) — rotate the GATED context
    /// (post-sigmoid-multiply), immediately before `o_proj`, using the
    /// CROSS-HEAD Hadamard factor (`H_num_heads ⊗ I_head_dim`, QuaRot Eq. 9
    /// "Hadamard heads" — mixes across heads at each fixed channel);
    /// `o_proj` absorbs the matching rotation on its input side. Expected to
    /// match dense output exactly (a pure post-nonlinearity linear
    /// reparameterization, structurally identical to the already-proven
    /// `rotation.rs` input-side absorption identity — the orientation
    /// (post-gate, input-side on `o_proj`) is what this identity requires;
    /// the AXIS must independently match QuaRot's cross-head factor, which
    /// is what `r3_axis_discriminates_cross_head_from_intra_head` below
    /// checks directly since this cancellation alone cannot tell the axis
    /// apart from an intra-head `BlockHadamard`).
    fn candidate_post_gate_context(fx: &Fixture) -> Vec<f32> {
        let (q, gate) = project_q_and_gate(fx);
        let context = attention_context(fx, &q, &fx.v_cache);
        let mut gated: Vec<f32> = context
            .iter()
            .zip(gate.iter())
            .map(|(&c, &g)| c * sigmoid(g))
            .collect();

        let r = RandomizedHadamard::new(R_O_SEED, NUM_Q_HEADS).unwrap();
        apply_cross_head_rotation(&mut gated, NUM_Q_HEADS, HEAD_DIM, &r);
        let mut o_rot = fx.o_proj.clone();
        absorb_input_cross_head_rotation(&mut o_rot, HIDDEN, Q_DIM, NUM_Q_HEADS, HEAD_DIM, &r);
        matvec(&o_rot, HIDDEN, Q_DIM, &gated)
    }

    /// Candidate 3 — rotate BOTH the (pre-gate) context and `gate_z`
    /// jointly, then apply sigmoid to the ROTATED gate before multiplying —
    /// a naive attempt to "push the rotation earlier" symmetrically.
    /// Expected to diverge: `sigmoid` is nonlinear, so
    /// `sigmoid(R . gate_z) != R . sigmoid(gate_z)` in general.
    fn candidate_joint_pre_gate_through_sigmoid(fx: &Fixture) -> Vec<f32> {
        let (q, gate) = project_q_and_gate(fx);
        let context = attention_context(fx, &q, &fx.v_cache);
        let r = RandomizedHadamard::new(0xC3, NUM_Q_HEADS).unwrap();
        let mut context_rot = context.clone();
        apply_cross_head_rotation(&mut context_rot, NUM_Q_HEADS, HEAD_DIM, &r);
        let mut gate_rot = gate.clone();
        apply_cross_head_rotation(&mut gate_rot, NUM_Q_HEADS, HEAD_DIM, &r);
        let gated: Vec<f32> = context_rot
            .iter()
            .zip(gate_rot.iter())
            .map(|(&c, &g)| c * sigmoid(g))
            .collect();

        let mut o_rot = fx.o_proj.clone();
        absorb_input_cross_head_rotation(&mut o_rot, HIDDEN, Q_DIM, NUM_Q_HEADS, HEAD_DIM, &r);
        matvec(&o_rot, HIDDEN, Q_DIM, &gated)
    }

    /// Candidate 4 — value-cache round trip composed with the winning
    /// post-gate orientation: rotate V pre-cache (benefits a future KV-cache
    /// quantization PR), immediately UNDO that same rotation on `context`
    /// before gating (net no-op on the math), gate normally, then apply the
    /// winning post-gate rotation and absorb it into `o_proj`'s input side.
    /// Expected to match dense output exactly — demonstrates the design
    /// doc's "value-path companion" is compatible with the winning
    /// orientation, not a distinct competing one.
    fn candidate_value_round_trip_then_post_gate(fx: &Fixture) -> Vec<f32> {
        let r_v = BlockHadamard::new(R_V_SEED, KV_DIM, HEAD_DIM).unwrap();
        let mut v_rotated = fx.v_cache.clone();
        for v in v_rotated.iter_mut() {
            r_v.apply(v).unwrap();
        }
        let (q, gate) = project_q_and_gate(fx);
        let context_rot = attention_context(fx, &q, &v_rotated);

        let broadcast = KvBroadcastRotation::matching(R_V_SEED);
        let mut context_restored = context_rot.clone();
        broadcast.apply_inverse(&mut context_restored);

        let mut gated: Vec<f32> = context_restored
            .iter()
            .zip(gate.iter())
            .map(|(&c, &g)| c * sigmoid(g))
            .collect();

        let r_o = RandomizedHadamard::new(R_O_SEED, NUM_Q_HEADS).unwrap();
        apply_cross_head_rotation(&mut gated, NUM_Q_HEADS, HEAD_DIM, &r_o);
        let mut o_rot = fx.o_proj.clone();
        absorb_input_cross_head_rotation(&mut o_rot, HIDDEN, Q_DIM, NUM_Q_HEADS, HEAD_DIM, &r_o);
        matvec(&o_rot, HIDDEN, Q_DIM, &gated)
    }

    #[test]
    fn r3_orientation_enumeration_finds_exactly_one_reparameterization() {
        let fx = build_fixture();
        let dense = dense_forward(&fx);

        const TOLERANCE: f32 = 1e-4;
        let candidates: [(&str, Vec<f32>, bool); 4] = [
            (
                "pre_gate_value_only",
                candidate_pre_gate_value_only(&fx),
                false,
            ),
            (
                "post_gate_context (winning orientation)",
                candidate_post_gate_context(&fx),
                true,
            ),
            (
                "joint_pre_gate_through_sigmoid",
                candidate_joint_pre_gate_through_sigmoid(&fx),
                false,
            ),
            (
                "value_round_trip_then_post_gate",
                candidate_value_round_trip_then_post_gate(&fx),
                true,
            ),
        ];

        let mut matched: usize = 0;
        for (name, out, expected_match) in &candidates {
            let delta = max_abs_diff(&dense, out);
            let is_match = delta < TOLERANCE;
            eprintln!(
                "R3 orientation candidate {name:>45}: max_abs_diff={delta:.6} match={is_match}"
            );
            assert_eq!(
                is_match, *expected_match,
                "candidate {name}: expected match={expected_match}, got delta={delta}"
            );
            if is_match {
                matched += 1;
            }
        }

        // The two matching candidates encode the SAME orientation — post-gate
        // rotation, o_proj input-side absorption — not two different correct
        // answers; the value round trip is a no-op composition on top of the
        // one winning orientation.
        assert_eq!(
            matched, 2,
            "expected exactly the two same-orientation candidates to match"
        );

        // The artifact schema's recorded orientation must agree with the
        // enumeration result.
        let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
        let spec = OnlineRotationSpec::r3_full_attention(&cfg, 1, NUM_Q_HEADS).unwrap();
        assert_eq!(
            spec.side,
            AbsorptionSide::InputSide,
            "OnlineRotationSpec::r3_full_attention must record the winning \
             orientation proven by this enumeration"
        );
    }

    /// Builds the explicit `N x N` basis matrix for the CROSS-HEAD factor
    /// `H_num_heads ⊗ I_head_dim` (QuaRot Eq. 9's "Hadamard heads"): for each
    /// fixed within-head channel `d`, the `num_heads` values at that channel
    /// are mixed by the SAME (unsigned, deterministic) Walsh-Hadamard.
    fn cross_head_matrix(num_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let n = num_heads * head_dim;
        let mut matrix = vec![vec![0.0f32; n]; n];
        for j in 0..n {
            let mut e = vec![0.0f32; n];
            e[j] = 1.0;
            for d in 0..head_dim {
                let mut channel: Vec<f32> = (0..num_heads).map(|h| e[h * head_dim + d]).collect();
                crate::quant::quarot::hadamard::walsh_hadamard_orthonormal_in_place(&mut channel)
                    .unwrap();
                for (h, &v) in channel.iter().enumerate() {
                    e[h * head_dim + d] = v;
                }
            }
            for (i, &v) in e.iter().enumerate() {
                matrix[i][j] = v;
            }
        }
        matrix
    }

    /// Builds the explicit `N x N` basis matrix for the INTRA-HEAD factor
    /// `I_num_heads ⊗ H_head_dim` — the axis the PRE-FIX R3 contract and
    /// reference used (contiguous per-head `head_dim` blocks, each rotated
    /// independently; no mixing across heads at all).
    fn intra_head_matrix(num_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let n = num_heads * head_dim;
        let mut matrix = vec![vec![0.0f32; n]; n];
        for j in 0..n {
            let mut e = vec![0.0f32; n];
            e[j] = 1.0;
            for h in 0..num_heads {
                let slice = &mut e[h * head_dim..(h + 1) * head_dim];
                crate::quant::quarot::hadamard::walsh_hadamard_orthonormal_in_place(slice).unwrap();
            }
            for (i, &v) in e.iter().enumerate() {
                matrix[i][j] = v;
            }
        }
        matrix
    }

    /// Discriminating test: the
    /// `o_proj`-cancellation identity in
    /// `r3_orientation_enumeration_finds_exactly_one_reparameterization`
    /// passes for EITHER Kronecker axis (any orthogonal rotation counter-
    /// rotates cleanly through `o_proj`'s input-side absorption), so it
    /// cannot tell QuaRot's real cross-head R3 apart from the intra-head
    /// construction the pre-fix contract used. This test checks the AXIS
    /// directly, on a small hand-verifiable `num_heads=4, head_dim=4` case.
    #[test]
    fn r3_axis_discriminates_cross_head_from_intra_head() {
        const NH: usize = 4;
        const DH: usize = 4;
        const N: usize = NH * DH;

        // Hand-computed orthonormal order-4 Walsh-Hadamard: H4 = H2 ⊗ H2
        // (QuaRot paper Eq. 1).
        let h4: [[f32; NH]; NH] = [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ];
        // Hand-computed H_4 ⊗ I_4: entry [h_out*DH+d][h_in*DH+d] = h4[h_out][h_in],
        // zero whenever the two within-head channel indices differ.
        let mut hand_cross = vec![vec![0.0f32; N]; N];
        for h_out in 0..NH {
            for h_in in 0..NH {
                for d in 0..DH {
                    hand_cross[h_out * DH + d][h_in * DH + d] = h4[h_out][h_in];
                }
            }
        }

        let built_cross = cross_head_matrix(NH, DH);
        for i in 0..N {
            for j in 0..N {
                assert!(
                    (built_cross[i][j] - hand_cross[i][j]).abs() < 1e-6,
                    "cross-head matrix[{i}][{j}]: built={}, hand-computed H_4⊗I_4={}",
                    built_cross[i][j],
                    hand_cross[i][j]
                );
            }
        }

        // Discriminating assertion: the cross-head factor mixes values
        // ACROSS heads at a FIXED channel — head 0's channel-0 output must
        // depend on head 1's channel-0 input (nonzero off-head-block entry).
        assert!(
            built_cross[0][DH].abs() > 1e-6,
            "H_num_heads ⊗ I_head_dim must mix across heads at a fixed channel"
        );
        // ...and must NOT mix across different within-head channels.
        assert_eq!(
            built_cross[0][DH + 1],
            0.0,
            "H_num_heads ⊗ I_head_dim must not mix different within-head channels"
        );

        // Mutation-sensitive: the PRE-FIX construction (I_num_heads ⊗
        // H_head_dim, contiguous per-head blocks) is exactly block-diagonal
        // per head — the SAME discriminating assertion above is FALSE for
        // it. This is why the o_proj-cancellation test alone could not
        // catch the wrong axis: cancellation holds for either construction,
        // but only the cross-head one satisfies this off-head-block check.
        let built_intra = intra_head_matrix(NH, DH);
        assert_eq!(
            built_intra[0][DH], 0.0,
            "I_num_heads ⊗ H_head_dim (the pre-fix construction) must NOT mix \
             across heads — confirms the discriminating assertion above would \
             fail (mutation-sensitive) against the old construction"
        );
        let differing = (0..N)
            .flat_map(|i| (0..N).map(move |j| (i, j)))
            .filter(|&(i, j)| (built_cross[i][j] - built_intra[i][j]).abs() > 1e-6)
            .count();
        assert!(
            differing > 0,
            "cross-head and intra-head constructions must be genuinely different matrices"
        );
    }
}
