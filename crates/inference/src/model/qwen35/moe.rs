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

fn compute_router_probs(
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

fn select_top_k(scratch: &mut ForwardScratch, num_experts: usize, top_k: usize) {
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

fn renormalize_selected(scratch: &mut ForwardScratch, top_k: usize) {
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
