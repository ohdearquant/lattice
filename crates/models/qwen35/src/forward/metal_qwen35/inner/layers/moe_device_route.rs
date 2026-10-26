//! ADR-053 D3': MoE routing on the device for Eager expert storage.
//!
//! The router, top-k selection, renormalization and the shared-expert gate run
//! as kernels inside the token's command buffer and write a per-layer route
//! buffer; the expert GEMVs and accumulations read their offsets and weights
//! from it. All dispatches go into one serial compute encoder, so each kernel
//! starts after the previous one completed: the router and gate kernels read
//! the current layer's norm output, and the routed work reads what they wrote.
//! Nothing here reads device memory on the host.

use super::super::*;

/// Whether `moe` takes the device route for this token.
///
/// The route buffer exists only for Eager storage of a supported shape. Tests
/// that force an expert list need the host router that honours it.
pub(super) fn moe_device_route_applies(moe: &MoeMetalBuffers) -> bool {
    if moe.device_route.is_none() || !matches!(moe.routed, RoutedExpertStorage::Eager { .. }) {
        return false;
    }
    #[cfg(test)]
    if FORCED_MOE_EXPERTS_FOR_TEST.with(|forced| forced.borrow().is_some()) {
        return false;
    }
    true
}

/// `MoeRouteParams` in `qwen35.metal`.
#[repr(C)]
struct MoeRouteParams {
    hidden: u32,
    num_experts: u32,
    top_k: u32,
    inter: u32,
}

const WORD: u64 = std::mem::size_of::<u32>() as u64;

impl MetalQwen35State {
    /// Encode one MoE FFN step on the device route into an open encoder.
    ///
    /// # Preconditions
    /// - `moe_device_route_applies(moe)` holds.
    /// - The current layer's post-attention norm was encoded into `enc`, so
    ///   `activations.hidden` holds its output when the router kernel runs.
    ///
    /// # Postconditions
    /// - `activations.ffn_out` holds the MoE FFN output once the encoder's
    ///   command buffer completes.
    ///
    /// # Safety
    /// `moe` is owned by `self.engine.layer_weights` and outlives the command
    /// buffer. Every id the router kernel writes is in `[0, num_experts)` and
    /// every offset it derives was bounded against the expert buffers when the
    /// route buffer was built (`moe_device_route_buffer`).
    pub(super) unsafe fn encode_moe_ffn_device(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &MoeMetalBuffers,
        layer_idx: usize,
        token_idx: usize,
    ) {
        let (Some(route), RoutedExpertStorage::Eager { gate_up, down }) =
            (&moe.device_route, &moe.routed)
        else {
            unreachable!("encode_moe_ffn_device requires an Eager layer with a route buffer");
        };
        let hidden = moe.hidden;
        let inter = moe.inter;
        let shared_inter = moe.shared_inter;
        let top_k = moe.top_k;
        let hidden_u32 = hidden as u32;
        let inter_u32 = inter as u32;
        let shared_inter_u32 = shared_inter as u32;
        let k_words = top_k as u64;
        let wg = 256u64;
        let elementwise = |count: usize| {
            (
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            )
        };

        #[cfg(test)]
        MOE_ROUTER_RUNS_FOR_TEST.with(|c| c.set((c.get().0, c.get().1 + 1)));

        // Zero the accumulator.
        enc.set_compute_pipeline_state(&self.engine.pipelines.moe_zero_buf);
        enc.set_buffer(0, Some(&moe.scratch_out), 0);
        enc.set_bytes(1, 4, &hidden_u32 as *const u32 as *const _);
        let (grid, group) = elementwise(hidden);
        enc.dispatch_threads(grid, group);

        // Router: ids, weights and offsets into the route buffer.
        let params = MoeRouteParams {
            hidden: hidden_u32,
            num_experts: moe.num_experts as u32,
            top_k: top_k as u32,
            inter: inter_u32,
        };
        enc.set_compute_pipeline_state(&self.engine.pipelines.moe_router_topk);
        enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
        enc.set_buffer(1, Some(&moe.router_gate), 0);
        enc.set_buffer(2, Some(route), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<MoeRouteParams>() as u64,
            &params as *const MoeRouteParams as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

        // Shared-expert gate scalar into route word 5K.
        let shared_gate_off = 5 * k_words * WORD;
        enc.set_compute_pipeline_state(&self.engine.pipelines.moe_shared_gate_scalar);
        enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
        enc.set_buffer(1, Some(&moe.shared_expert_gate), 0);
        enc.set_buffer(2, Some(route), shared_gate_off);
        enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

        // Shared expert: gate and up GEMVs, SiLU-mul, down GEMV, gated accumulate.
        let gemv = |x: &Buffer, w: &Buffer, out: &Buffer, n: u32, k: u32| {
            let params = GemmParams {
                m: 1,
                n,
                k,
                lda: k,
                ldb: k,
                ldc: n,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(out), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params as *const GemmParams as *const _,
            );
            enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        };
        let silu_mul = |count: usize| {
            let count_u32 = count as u32;
            enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
            enc.set_buffer(0, Some(&moe.scratch_gate), 0);
            enc.set_buffer(1, Some(&moe.scratch_up), 0);
            enc.set_bytes(2, 4, &count_u32 as *const u32 as *const _);
            let (grid, group) = elementwise(count);
            enc.dispatch_threads(grid, group);
        };
        let scale_add = |scale_off: u64| {
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_scale_add_routed);
            enc.set_buffer(0, Some(&moe.scratch_out), 0);
            enc.set_buffer(1, Some(&moe.scratch_expert_out), 0);
            enc.set_buffer(2, Some(route), scale_off);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
            let (grid, group) = elementwise(hidden);
            enc.dispatch_threads(grid, group);
        };
        let hidden_act = &self.session.activations.hidden;
        gemv(
            hidden_act,
            &moe.shared_gate_proj,
            &moe.scratch_gate,
            shared_inter_u32,
            hidden_u32,
        );
        gemv(
            hidden_act,
            &moe.shared_up_proj,
            &moe.scratch_up,
            shared_inter_u32,
            hidden_u32,
        );
        silu_mul(shared_inter);
        gemv(
            &moe.scratch_gate,
            &moe.shared_down_proj,
            &moe.scratch_expert_out,
            hidden_u32,
            shared_inter_u32,
        );
        scale_add(shared_gate_off);

        // Routed experts, rank order: offsets and weights from the route buffer.
        let routed_gemv = |x: &Buffer, w: &Buffer, out: &Buffer, n: u32, k: u32, off_word: u64| {
            let params = GemmParams {
                m: 1,
                n,
                k,
                lda: k,
                ldb: k,
                ldc: n,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_expert_gemv_routed);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(out), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params as *const GemmParams as *const _,
            );
            enc.set_buffer(4, Some(route), off_word * WORD);
            enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        };
        for rank in 0..k_words {
            let offsets = 2 * k_words + 3 * rank;
            routed_gemv(
                hidden_act,
                gate_up,
                &moe.scratch_gate,
                inter_u32,
                hidden_u32,
                offsets,
            );
            routed_gemv(
                hidden_act,
                gate_up,
                &moe.scratch_up,
                inter_u32,
                hidden_u32,
                offsets + 1,
            );
            silu_mul(inter);
            routed_gemv(
                &moe.scratch_gate,
                down,
                &moe.scratch_expert_out,
                hidden_u32,
                inter_u32,
                offsets + 2,
            );
            scale_add((k_words + rank) * WORD);
        }

        // Accumulator into ffn_out.
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy);
        enc.set_buffer(0, Some(&moe.scratch_out), 0);
        enc.set_buffer(1, Some(&self.session.activations.ffn_out), 0);
        enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const _);
        let (grid, group) = elementwise(hidden);
        enc.dispatch_threads(grid, group);

        MOE_ROUTING_TRACE.with(|c| {
            if c.borrow().is_some() {
                MOE_DEVICE_ROUTE_PENDING.with(|p| {
                    p.borrow_mut()
                        .push((layer_idx, token_idx, top_k, route.clone()))
                });
            }
        });
    }
}
