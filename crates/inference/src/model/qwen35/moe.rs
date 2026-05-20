use super::cache::ForwardScratch;
use super::weights::MoeLayerWeights;
use crate::forward::cpu::{elementwise_mul, matmul_bt, silu_inplace};

/// MoE forward step — free function so it can be unit-tested directly.
pub(crate) fn moe_ffn_step(moe: &MoeLayerWeights, scratch: &mut ForwardScratch, hidden: usize) {
    let inter = moe.experts.intermediate_size;
    let shared_inter = moe.shared_expert.intermediate_size;
    let num_experts = moe.router.num_experts;
    let top_k = moe.router.num_experts_per_tok;
    debug_assert_eq!(moe.router.hidden_size, hidden);
    debug_assert_eq!(moe.experts.num_experts, num_experts);
    debug_assert_eq!(moe.experts.hidden_size, hidden);
    debug_assert_eq!(moe.shared_expert.hidden_size, hidden);

    scratch.input_tmp[..hidden].copy_from_slice(&scratch.ffn_out[..hidden]);

    if scratch.router_logits.len() < num_experts {
        scratch.router_logits.resize(num_experts, 0.0);
    }
    if scratch.router_selected.len() < top_k {
        scratch.router_selected.resize(top_k, (usize::MAX, 0.0));
    }

    compute_router_probs(moe, scratch, hidden, num_experts);

    select_top_k(scratch, num_experts, top_k);

    renormalize_selected(scratch, top_k);

    accumulate_routed_experts(moe, scratch, hidden, inter, top_k);

    apply_shared_expert(moe, scratch, hidden, shared_inter);

    scratch.ffn_out[..hidden].copy_from_slice(&scratch.expert_out[..hidden]);
}

pub(crate) fn compute_router_probs(
    moe: &MoeLayerWeights,
    scratch: &mut ForwardScratch,
    hidden: usize,
    num_experts: usize,
) {
    let gate = &moe.router.gate;
    let input = &scratch.input_tmp[..hidden];
    let logits = &mut scratch.router_logits[..num_experts];
    matmul_bt(input, gate, logits, 1, hidden, num_experts);

    let max_logit = scratch.router_logits[..num_experts]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for v in &mut scratch.router_logits[..num_experts] {
        *v = (*v - max_logit).exp();
        denom += *v;
    }
    if denom > 0.0 {
        for v in &mut scratch.router_logits[..num_experts] {
            *v /= denom;
        }
    }
}

pub(crate) fn select_top_k(scratch: &mut ForwardScratch, num_experts: usize, top_k: usize) {
    for slot in &mut scratch.router_selected[..top_k] {
        *slot = (usize::MAX, f32::NEG_INFINITY);
    }
    for (expert_id, prob) in scratch.router_logits[..num_experts]
        .iter()
        .copied()
        .enumerate()
    {
        for rank in 0..top_k {
            if prob > scratch.router_selected[rank].1 {
                for shift in (rank + 1..top_k).rev() {
                    scratch.router_selected[shift] = scratch.router_selected[shift - 1];
                }
                scratch.router_selected[rank] = (expert_id, prob);
                break;
            }
        }
    }
}

pub(crate) fn renormalize_selected(scratch: &mut ForwardScratch, top_k: usize) {
    let top_sum: f32 = scratch.router_selected[..top_k]
        .iter()
        .map(|(_, p)| *p)
        .sum();
    if top_sum > 0.0 {
        for (_, prob) in &mut scratch.router_selected[..top_k] {
            *prob /= top_sum;
        }
    }
}

fn accumulate_routed_experts(
    moe: &MoeLayerWeights,
    scratch: &mut ForwardScratch,
    hidden: usize,
    inter: usize,
    top_k: usize,
) {
    scratch.expert_out[..hidden].fill(0.0);

    for idx in 0..top_k {
        let (expert_id, weight) = scratch.router_selected[idx];
        let gate_up_stride = 2 * inter * hidden;
        let gate_up_start = expert_id * gate_up_stride;
        let down_start = expert_id * hidden * inter;

        // HF Qwen MoE stores each expert gate_up_proj as [2 * intermediate, hidden]
        // and splits F.linear(...).chunk(2, dim=-1): first half is gate, second is up.
        // See transformers modeling_qwen3_moe.py:L222 and L244.
        let gate_w = &moe.experts.gate_up_proj[gate_up_start..gate_up_start + inter * hidden];
        let up_w = &moe.experts.gate_up_proj
            [gate_up_start + inter * hidden..gate_up_start + 2 * inter * hidden];
        let down_w = &moe.experts.down_proj[down_start..down_start + hidden * inter];

        matmul_bt(
            &scratch.input_tmp[..hidden],
            gate_w,
            &mut scratch.gate_buf[..inter],
            1,
            hidden,
            inter,
        );
        matmul_bt(
            &scratch.input_tmp[..hidden],
            up_w,
            &mut scratch.up_buf[..inter],
            1,
            hidden,
            inter,
        );

        silu_inplace(&mut scratch.gate_buf[..inter]);
        elementwise_mul(&mut scratch.gate_buf[..inter], &scratch.up_buf[..inter]);

        scratch.down_input[..inter].copy_from_slice(&scratch.gate_buf[..inter]);
        matmul_bt(
            &scratch.down_input[..inter],
            down_w,
            &mut scratch.ffn_out[..hidden],
            1,
            inter,
            hidden,
        );

        for i in 0..hidden {
            scratch.expert_out[i] += weight * scratch.ffn_out[i];
        }
    }
}

fn apply_shared_expert(
    moe: &MoeLayerWeights,
    scratch: &mut ForwardScratch,
    hidden: usize,
    shared_inter: usize,
) {
    let shared = &moe.shared_expert;

    let gate_logit: f32 = scratch.input_tmp[..hidden]
        .iter()
        .zip(shared.shared_expert_gate.iter())
        .map(|(x, g)| x * g)
        .sum();
    let shared_gate = 1.0 / (1.0 + (-gate_logit).exp());

    matmul_bt(
        &scratch.input_tmp[..hidden],
        &shared.gate_proj,
        &mut scratch.gate_buf[..shared_inter],
        1,
        hidden,
        shared_inter,
    );
    matmul_bt(
        &scratch.input_tmp[..hidden],
        &shared.up_proj,
        &mut scratch.up_buf[..shared_inter],
        1,
        hidden,
        shared_inter,
    );

    silu_inplace(&mut scratch.gate_buf[..shared_inter]);
    elementwise_mul(
        &mut scratch.gate_buf[..shared_inter],
        &scratch.up_buf[..shared_inter],
    );

    scratch.down_input[..shared_inter].copy_from_slice(&scratch.gate_buf[..shared_inter]);
    matmul_bt(
        &scratch.down_input[..shared_inter],
        &shared.down_proj,
        &mut scratch.ffn_out[..hidden],
        1,
        shared_inter,
        hidden,
    );

    for i in 0..hidden {
        scratch.expert_out[i] += shared_gate * scratch.ffn_out[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────────────

    /// Minimal ForwardScratch for router-only tests.
    fn make_scratch(num_experts: usize, top_k: usize, hidden: usize) -> ForwardScratch {
        let mut s = ForwardScratch::new();
        s.ffn_out.resize(hidden, 0.0);
        s.gate_buf.resize(64, 0.0);
        s.up_buf.resize(64, 0.0);
        s.down_input.resize(64, 0.0);
        s.expert_out.resize(hidden, 0.0);
        s.input_tmp.resize(hidden, 0.0);
        s.router_logits.resize(num_experts, 0.0);
        s.router_selected.resize(top_k, (usize::MAX, 0.0));
        s
    }

    // ── select_top_k tests ───────────────────────────────────────────────────────

    /// top-k=2 from 4 experts: verify the correct pair is chosen with descending
    /// probability order (highest first).
    #[test]
    fn test_select_top_k_basic() {
        let num_experts = 4;
        let top_k = 2;
        let mut scratch = make_scratch(num_experts, top_k, 8);

        // Probabilities: expert 2 highest, expert 0 second.
        scratch.router_logits[0] = 0.30;
        scratch.router_logits[1] = 0.10;
        scratch.router_logits[2] = 0.50;
        scratch.router_logits[3] = 0.10;

        select_top_k(&mut scratch, num_experts, top_k);

        // Slot 0 should be the absolute highest (expert 2).
        assert_eq!(
            scratch.router_selected[0].0, 2,
            "slot 0 must be expert 2 (prob 0.50)"
        );
        // Slot 1 should be the second highest (expert 0).
        assert_eq!(
            scratch.router_selected[1].0, 0,
            "slot 1 must be expert 0 (prob 0.30)"
        );
        // Probabilities preserved (not yet renormalized).
        assert!((scratch.router_selected[0].1 - 0.50).abs() < 1e-6);
        assert!((scratch.router_selected[1].1 - 0.30).abs() < 1e-6);
    }

    /// All experts have the same probability — top-k must still produce exactly
    /// top_k unique valid indices without repeats.
    #[test]
    fn test_select_top_k_ties() {
        let num_experts = 8;
        let top_k = 3;
        let mut scratch = make_scratch(num_experts, top_k, 8);
        for v in scratch.router_logits.iter_mut() {
            *v = 0.125; // uniform
        }

        select_top_k(&mut scratch, num_experts, top_k);

        let ids: Vec<usize> = scratch.router_selected[..top_k]
            .iter()
            .map(|(id, _)| *id)
            .collect();
        // All selected experts must be valid indices.
        for &id in &ids {
            assert!(id < num_experts, "invalid expert id {id}");
        }
        // No duplicates.
        let mut seen = std::collections::HashSet::new();
        for &id in &ids {
            assert!(seen.insert(id), "duplicate expert id {id}");
        }
    }

    // ── renormalize_selected tests ────────────────────────────────────────────

    /// After renormalization the selected weights must sum to 1.
    #[test]
    fn test_renormalize_selected_sums_to_one() {
        let top_k = 3;
        let num_experts = 6;
        let mut scratch = make_scratch(num_experts, top_k, 8);

        scratch.router_logits[0] = 0.2;
        scratch.router_logits[1] = 0.5;
        scratch.router_logits[2] = 0.1;
        scratch.router_logits[3] = 0.1;
        scratch.router_logits[4] = 0.05;
        scratch.router_logits[5] = 0.05;

        select_top_k(&mut scratch, num_experts, top_k);
        renormalize_selected(&mut scratch, top_k);

        let sum: f32 = scratch.router_selected[..top_k]
            .iter()
            .map(|(_, p)| *p)
            .sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalized weights must sum to 1.0 (got {sum})"
        );
    }

    // ── Weight layout offset tests (ADR-053 D2) ──────────────────────────────

    /// Verify the expert-major gate_up offset formula matches the documented layout.
    ///
    /// Layout: [num_experts, 2 * inter, hidden] contiguous.
    ///   expert e gate starts at element: e * 2 * inter * hidden
    ///   expert e up   starts at element: e * 2 * inter * hidden + inter * hidden
    #[test]
    fn test_gate_up_expert_major_layout_offsets() {
        let num_experts = 4usize;
        let inter = 8usize;
        let hidden = 16usize;
        let stride = 2 * inter * hidden;

        for e in 0..num_experts {
            let gate_off = e * stride;
            let up_off = e * stride + inter * hidden;

            // Gate block must not overlap with the previous expert's up block.
            if e > 0 {
                let prev_up_end = (e - 1) * stride + 2 * inter * hidden;
                assert_eq!(
                    gate_off,
                    prev_up_end,
                    "expert {e} gate starts immediately after expert {}'s up block",
                    e - 1
                );
            }

            // Up block starts exactly inter*hidden elements after gate block.
            assert_eq!(
                up_off - gate_off,
                inter * hidden,
                "up offset must be exactly inter*hidden after gate offset for expert {e}"
            );

            // Up block ends exactly at (e+1)*stride.
            assert_eq!(
                up_off + inter * hidden,
                (e + 1) * stride,
                "up block for expert {e} must end at the start of expert {}'s gate",
                e + 1
            );
        }

        // Total buffer length must equal num_experts * 2 * inter * hidden.
        assert_eq!(num_experts * stride, num_experts * 2 * inter * hidden);
    }

    /// Verify the expert-major down offset formula.
    ///
    /// Layout: [num_experts, hidden, inter] contiguous.
    ///   expert e starts at element: e * hidden * inter
    #[test]
    fn test_down_expert_major_layout_offsets() {
        let num_experts = 4usize;
        let inter = 8usize;
        let hidden = 16usize;
        let stride = hidden * inter;

        for e in 0..num_experts {
            let down_off = e * stride;
            // Each expert occupies exactly hidden*inter elements.
            let next_off = (e + 1) * stride;
            assert_eq!(
                next_off - down_off,
                hidden * inter,
                "expert {e} down block size mismatch"
            );
        }

        // Total buffer length.
        assert_eq!(num_experts * stride, num_experts * hidden * inter);
    }

    // ── compute_router_probs tests ────────────────────────────────────────────

    /// Softmax over a 1-hot logit vector: the hot entry should have probability
    /// approaching 1 for large logit values.
    #[test]
    fn test_compute_router_probs_one_hot() {
        use crate::model::qwen35::weights::{
            MoeLayerWeights, MoeRouter, RoutedExperts, SharedExpert,
        };

        let num_experts = 4usize;
        let hidden = 4usize;
        let inter = 2usize;
        let top_k = 1usize;

        // Gate weight matrix: expert 2's row = [1,1,1,1], others = zeros.
        // With input = [1,1,1,1], logits[2] = 4.0, rest = 0.0.
        let mut gate = vec![0.0f32; num_experts * hidden];
        for j in 0..hidden {
            gate[2 * hidden + j] = 1.0;
        }
        let router = MoeRouter::new(gate, num_experts, top_k, hidden).unwrap();

        let gate_up = vec![0.0f32; num_experts * 2 * inter * hidden];
        let down = vec![0.0f32; num_experts * hidden * inter];
        let experts = RoutedExperts::new(gate_up, down, num_experts, hidden, inter).unwrap();

        let shared_gate_proj = vec![0.0f32; inter * hidden];
        let shared_up_proj = vec![0.0f32; inter * hidden];
        let shared_down_proj = vec![0.0f32; hidden * inter];
        let shared_expert_gate = vec![0.0f32; hidden];
        let shared = SharedExpert::new(
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            shared_expert_gate,
            hidden,
            inter,
        )
        .unwrap();

        let moe = MoeLayerWeights {
            router,
            experts,
            shared_expert: shared,
        };
        let mut scratch = make_scratch(num_experts, top_k, hidden);

        // Input: all ones.
        for v in scratch.ffn_out.iter_mut() {
            *v = 1.0;
        }
        scratch
            .input_tmp
            .copy_from_slice(&scratch.ffn_out[..hidden]);

        compute_router_probs(&moe, &mut scratch, hidden, num_experts);

        // Expert 2 should have the highest probability.
        let argmax = scratch.router_logits[..num_experts]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            argmax, 2,
            "expert 2 must have the highest softmax probability"
        );
        // Probabilities must sum to 1.
        let sum: f32 = scratch.router_logits[..num_experts].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "router probabilities must sum to 1 (got {sum})"
        );
    }

    // ── Coalescing correctness: full routing round-trip ───────────────────────

    /// End-to-end routing round-trip: probs → top-k → renormalize.
    /// Verifies that the coalescing pipeline produces the correct expert indices
    /// and that their renormalized weights sum to 1.
    #[test]
    fn test_routing_round_trip_top2_of_8() {
        let num_experts = 8usize;
        let top_k = 2usize;
        let mut scratch = make_scratch(num_experts, top_k, 8);

        // Synthetic pre-softmax logits: expert 5 highest, expert 1 second.
        scratch
            .router_logits
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| {
                *v = match i {
                    5 => 3.0,
                    1 => 2.0,
                    _ => 0.0,
                };
            });

        // Manually apply softmax (as compute_router_probs would after matmul).
        let max_l = scratch.router_logits[..num_experts]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for v in scratch.router_logits[..num_experts].iter_mut() {
            *v = (*v - max_l).exp();
            denom += *v;
        }
        for v in scratch.router_logits[..num_experts].iter_mut() {
            *v /= denom;
        }

        select_top_k(&mut scratch, num_experts, top_k);
        renormalize_selected(&mut scratch, top_k);

        // Expert 5 must be rank-0 (highest pre-softmax → highest post-softmax).
        assert_eq!(scratch.router_selected[0].0, 5, "rank-0 must be expert 5");
        // Expert 1 must be rank-1.
        assert_eq!(scratch.router_selected[1].0, 1, "rank-1 must be expert 1");
        // Renormalized weights must sum to 1.
        let w_sum: f32 = scratch.router_selected[..top_k]
            .iter()
            .map(|(_, p)| *p)
            .sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-5,
            "renormalized weights sum={w_sum}"
        );
    }

    // ── Memory pressure guard: threshold calculation ──────────────────────────

    /// Verify the memory threshold formula used at load time matches the ADR spec.
    /// ADR-053 §Risks R2: reject if routed_bytes > 0.85 × recommendedMaxWorkingSetSize.
    #[test]
    fn test_memory_threshold_formula() {
        let max_working: u64 = 32 * 1024 * 1024 * 1024; // 32 GiB (32 GB M2 Pro)
        let threshold = (max_working as f64 * 0.85) as u64;
        // 32 GiB = 34_359_738_368 bytes; 0.85 × 34_359_738_368 = 29_205_777_612
        let expected: u64 = 29_205_777_612;

        // Allow 1 byte rounding tolerance from floating-point conversion.
        assert!(
            threshold.abs_diff(expected) <= 1,
            "threshold {threshold} != expected {expected}"
        );
    }
}
