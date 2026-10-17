use super::scalar::ScalarMath;
use super::{Completion, ForcedExecutionError, Operation};
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::attention::gdn_backward::{GdnGrads, GdnSaved, gdn_backward_with};
use crate::backward::attention_gqa::{AttnCache, AttnGrads, gqa_backward_with};

fn require(
    operation: Operation,
    valid: bool,
    problem: &'static str,
) -> Result<(), ForcedExecutionError> {
    if valid {
        Ok(())
    } else {
        Err(ForcedExecutionError::Shape { operation, problem })
    }
}

fn size(operation: Operation, dimensions: &[usize]) -> Result<usize, ForcedExecutionError> {
    let elements = dimensions.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    match elements {
        Some(n)
            if n.checked_mul(size_of::<f32>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize) =>
        {
            Ok(n)
        }
        _ => Err(ForcedExecutionError::Shape {
            operation,
            problem: "shape overflow",
        }),
    }
}

fn lengths(operation: Operation, entries: &[(&[f32], usize)]) -> Result<(), ForcedExecutionError> {
    for &(values, expected) in entries {
        require(operation, values.len() == expected, "buffer length")?;
    }
    Ok(())
}

pub(super) fn gqa_backward(
    completion: Completion<'_>,
    dy_out: &[f32],
    cache: &AttnCache,
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    lora_a_q: Option<&[f32]>,
    lora_b_q: Option<&[f32]>,
    lora_a_v: Option<&[f32]>,
    lora_b_v: Option<&[f32]>,
    lora_rank: usize,
    lora_scale: f32,
) -> Result<AttnGrads, ForcedExecutionError> {
    let op = Operation::GqaBackward;
    let t = cache.seq_len;
    require(
        op,
        t > 0,
        "GQA sequence must be nonempty to infer hidden size",
    )?;
    require(op, cache.x_input.len().is_multiple_of(t), "input rows")?;
    require(op, cache.num_kv_heads > 0, "KV heads must be nonzero")?;
    require(
        op,
        cache.num_q_heads.is_multiple_of(cache.num_kv_heads),
        "head grouping",
    )?;
    require(op, cache.head_dim > 0, "head dimension must be nonzero")?;
    require(
        op,
        cache.rope_dim <= cache.head_dim && cache.rope_dim.is_multiple_of(2),
        "RoPE dimension",
    )?;
    let hidden = cache.x_input.len() / t;
    let q = size(op, &[cache.num_q_heads, cache.head_dim])?;
    let kv = size(op, &[cache.num_kv_heads, cache.head_dim])?;
    let twice_q = size(op, &[2, q])?;
    let th = size(op, &[t, hidden])?;
    let tq = size(op, &[t, q])?;
    let tkv = size(op, &[t, kv])?;
    let tr = size(op, &[t, lora_rank])?;
    let rh = size(op, &[lora_rank, hidden])?;
    let qr = size(op, &[twice_q, lora_rank])?;
    let vr = size(op, &[kv, lora_rank])?;
    lengths(
        op,
        &[
            (dy_out, th),
            (w_q, size(op, &[twice_q, hidden])?),
            (w_k, size(op, &[kv, hidden])?),
            (w_v, size(op, &[kv, hidden])?),
            (w_o, size(op, &[hidden, q])?),
            (q_norm_w, cache.head_dim),
            (k_norm_w, cache.head_dim),
            (&cache.q_raw, tq),
            (&cache.k_raw, tkv),
            (&cache.q_pre_rope, tq),
            (&cache.k_pre_rope, tkv),
            (&cache.v, tkv),
            (&cache.context, tq),
            (&cache.gate_z, tq),
            (&cache.h_q, tr),
            (&cache.h_v, tr),
        ],
    )?;
    require(
        op,
        cache.q_h.len() == t
            && cache.softmax_probs.len() == t
            && cache.cos_vals.len() == t
            && cache.sin_vals.len() == t,
        "cache row count",
    )?;
    for row in 0..t {
        lengths(
            op,
            &[
                (&cache.q_h[row], q),
                (
                    &cache.softmax_probs[row],
                    size(op, &[cache.num_q_heads, row + 1])?,
                ),
                (&cache.cos_vals[row], cache.rope_dim / 2),
                (&cache.sin_vals[row], cache.rope_dim / 2),
            ],
        )?;
    }
    for (a, b, b_len) in [(lora_a_q, lora_b_q, qr), (lora_a_v, lora_b_v, vr)] {
        match (a, b) {
            (Some(a), Some(b)) => lengths(op, &[(a, rh), (b, b_len)])?,
            (None, None) => {}
            _ => {
                return Err(ForcedExecutionError::Shape {
                    operation: op,
                    problem: "incomplete LoRA pair",
                });
            }
        }
    }
    let result = gqa_backward_with::<ScalarMath>(
        dy_out, cache, w_q, w_k, w_v, w_o, q_norm_w, k_norm_w, lora_a_q, lora_b_q, lora_a_v,
        lora_b_v, lora_rank, lora_scale,
    );
    completion.finish(op)?;
    Ok(result)
}

pub(super) fn gdn_backward(
    completion: Completion<'_>,
    grad_outputs: &[f32],
    saved: &GdnSaved,
    weights: &GatedDeltaNetWeights,
) -> Result<GdnGrads, ForcedExecutionError> {
    let op = Operation::GdnBackward;
    let t = saved.seq_len;
    let h = saved.hidden_size;
    let kh = saved.num_key_heads;
    let vh = saved.num_value_heads;
    let k = saved.key_dim;
    let v = saved.value_dim;
    require(
        op,
        kh > 0 && vh > 0 && k > 0 && v > 0,
        "head dimensions must be nonzero",
    )?;
    require(
        op,
        vh.is_multiple_of(kh) && saved.ratio == vh / kh,
        "head grouping",
    )?;
    require(
        op,
        saved.kernel_size > 0,
        "convolution kernel must be nonzero",
    )?;
    let output = size(op, &[vh, v])?;
    let qk = size(op, &[2, kh, k])?;
    let qkv = qk.checked_add(output).ok_or(ForcedExecutionError::Shape {
        operation: op,
        problem: "QKV dimension overflow",
    })?;
    require(
        op,
        saved.qkv_dim == qkv && saved.output_dim == output,
        "projection dimensions",
    )?;
    let th = size(op, &[t, h])?;
    let tqkv = size(op, &[t, qkv])?;
    let tout = size(op, &[t, output])?;
    let tvh = size(op, &[t, vh])?;
    let tvhk = size(op, &[t, vh, k])?;
    let state = size(op, &[vh, k, v])?;
    let tstate = size(op, &[t, state])?;
    let conv = size(op, &[t, qkv, saved.kernel_size - 1])?;
    lengths(
        op,
        &[
            (grad_outputs, th),
            (&saved.inputs, th),
            (&saved.qkv_proj, tqkv),
            (&saved.z_proj, tout),
            (&saved.beta_raw, tvh),
            (&saved.alpha_proj, tvh),
            (&saved.beta, tvh),
            (&saved.g, tvh),
            (&saved.conv_out, tqkv),
            (&saved.conv_buffers, conv),
            (&saved.q_hat, tvhk),
            (&saved.k_hat, tvhk),
            (&saved.v, tout),
            (&saved.q_norm, tvh),
            (&saved.k_norm, tvh),
            (&saved.q_eps_norm, tvh),
            (&saved.k_eps_norm, tvh),
            (&saved.kv_mem, tout),
            (&saved.s_after, tstate),
            (&saved.o_heads, tout),
            (&saved.rms_vals, tvh),
            (&saved.silu_z, tout),
            (&weights.in_proj_qkv, size(op, &[qkv, h])?),
            (&weights.in_proj_z, size(op, &[output, h])?),
            (&weights.in_proj_a, size(op, &[vh, h])?),
            (&weights.in_proj_b, size(op, &[vh, h])?),
            (&weights.out_proj, size(op, &[h, output])?),
            (&weights.conv1d_weight, size(op, &[qkv, saved.kernel_size])?),
            (&weights.norm_weight, v),
            (&weights.a_log, vh),
            (&weights.dt_bias, vh),
        ],
    )?;
    require(
        op,
        weights.in_proj_qkv_rows == qkv
            && weights.in_proj_qkv_cols == h
            && weights.in_proj_z_rows == output
            && weights.in_proj_z_cols == h
            && weights.in_proj_b_rows == vh
            && weights.in_proj_b_cols == h
            && weights.in_proj_a_rows == vh
            && weights.in_proj_a_cols == h
            && weights.out_proj_rows == h
            && weights.out_proj_cols == output
            && weights.conv_dim == qkv
            && weights.kernel_size == saved.kernel_size,
        "weight dimensions",
    )?;
    let rank = saved.lora_rank;
    let tr = size(op, &[t, rank])?;
    for activation in [
        &saved.h_qkv,
        &saved.h_z,
        &saved.h_b,
        &saved.h_a,
        &saved.h_out,
    ] {
        lengths(op, &[(activation, tr)])?;
    }
    lengths(op, &[(&saved.gated_buf, if rank > 0 { tout } else { 0 })])?;
    for (a, b, input, output) in [
        (&saved.lora_a_qkv, &saved.lora_b_qkv, h, qkv),
        (&saved.lora_a_z, &saved.lora_b_z, h, output),
        (&saved.lora_a_b, &saved.lora_b_b, h, vh),
        (&saved.lora_a_a, &saved.lora_b_a, h, vh),
        (&saved.lora_a_out, &saved.lora_b_out, output, h),
    ] {
        lengths(
            op,
            &[
                (a, size(op, &[rank, input])?),
                (b, size(op, &[output, rank])?),
            ],
        )?;
    }
    let result = gdn_backward_with::<ScalarMath>(grad_outputs, saved, weights);
    completion.finish(op)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::gdn_backward::{gdn_backward as automatic_gdn, gdn_forward_save};
    use crate::backward::attention_gqa::{gqa_backward as automatic_gqa, gqa_forward_with_cache};
    use crate::backward::simd::{Backend, ForcedDispatch, force_scalar};
    use crate::forward::cpu::observe_matmul_calls;
    use crate::model::qwen35_config::Qwen35Config;

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + seed * 13) % 47) as f32 - 23.0) * 0.007)
            .collect()
    }

    fn parity(expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len());
        let mut difference = 0.0f64;
        let mut magnitude = 0.0f64;
        for (&a, &b) in expected.iter().zip(actual) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else if a.is_infinite() {
                assert_eq!(a, b);
            } else {
                assert!(b.is_finite(), "finite reference {a}, forced {b}");
                difference += (f64::from(a) - f64::from(b)).powi(2);
                magnitude += f64::from(a).powi(2) + f64::from(b).powi(2);
            }
        }
        assert!((2.0 * difference / magnitude.max(1e-30)).sqrt() < 1e-4);
    }

    fn gqa_arrays(grads: &AttnGrads) -> [&[f32]; 5] {
        [
            &grads.grad_a_q,
            &grads.grad_b_q,
            &grads.grad_a_v,
            &grads.grad_b_v,
            &grads.dx,
        ]
    }

    fn gdn_arrays(grads: &GdnGrads) -> [&[f32]; 11] {
        [
            &grads.grad_a_qkv,
            &grads.grad_b_qkv,
            &grads.grad_a_z,
            &grads.grad_b_z,
            &grads.grad_a_b,
            &grads.grad_b_b,
            &grads.grad_a_a,
            &grads.grad_b_a,
            &grads.grad_a_out,
            &grads.grad_b_out,
            &grads.dx,
        ]
    }

    struct GqaFixture {
        cache: AttnCache,
        dy: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        norm: Vec<f32>,
        a_q: Vec<f32>,
        b_q: Vec<f32>,
        a_v: Vec<f32>,
        b_v: Vec<f32>,
        q_lora: bool,
        v_lora: bool,
        rank: usize,
    }

    impl GqaFixture {
        fn new(q_lora: bool, v_lora: bool, rank: usize) -> Self {
            let (t, hidden, q_heads, kv_heads, head, rope) = (3, 7, 4, 2, 3, 2);
            let q = q_heads * head;
            let kv = kv_heads * head;
            let w_q = values(2 * q * hidden, 1);
            let w_k = values(kv * hidden, 2);
            let w_v = values(kv * hidden, 3);
            let w_o = values(hidden * q, 4);
            let norm = vec![0.3; head];
            let a_q = values(rank * hidden, 5);
            let b_q = values(2 * q * rank, 6);
            let a_v = values(rank * hidden, 7);
            let b_v = values(kv * rank, 8);
            let (_, cache) = gqa_forward_with_cache(
                &values(t * hidden, 9),
                &w_q,
                &w_k,
                &w_v,
                &w_o,
                &norm,
                &norm,
                q_lora.then_some(a_q.as_slice()),
                q_lora.then_some(b_q.as_slice()),
                v_lora.then_some(a_v.as_slice()),
                v_lora.then_some(b_v.as_slice()),
                rank,
                1.7,
                t,
                hidden,
                q_heads,
                kv_heads,
                head,
                rope,
                &[1.0, 0.8, 0.6],
                &[0.0, 0.6, 0.8],
                1e-6,
            );
            Self {
                cache,
                dy: values(t * hidden, 10),
                w_q,
                w_k,
                w_v,
                w_o,
                norm,
                a_q,
                b_q,
                a_v,
                b_v,
                q_lora,
                v_lora,
                rank,
            }
        }

        fn ordinary(&self) -> AttnGrads {
            automatic_gqa(
                &self.dy,
                &self.cache,
                &self.w_q,
                &self.w_k,
                &self.w_v,
                &self.w_o,
                &self.norm,
                &self.norm,
                self.q_lora.then_some(self.a_q.as_slice()),
                self.q_lora.then_some(self.b_q.as_slice()),
                self.v_lora.then_some(self.a_v.as_slice()),
                self.v_lora.then_some(self.b_v.as_slice()),
                self.rank,
                1.7,
            )
        }

        fn forced(&self, dispatch: &ForcedDispatch) -> Result<AttnGrads, ForcedExecutionError> {
            dispatch.gqa_backward(
                &self.dy,
                &self.cache,
                &self.w_q,
                &self.w_k,
                &self.w_v,
                &self.w_o,
                &self.norm,
                &self.norm,
                self.q_lora.then_some(self.a_q.as_slice()),
                self.q_lora.then_some(self.b_q.as_slice()),
                self.v_lora.then_some(self.a_v.as_slice()),
                self.v_lora.then_some(self.b_v.as_slice()),
                self.rank,
                1.7,
            )
        }
    }

    #[test]
    fn forced_gqa_all_arrays_and_lora_permutations_bypass_gemm() {
        for (q_lora, v_lora, rank) in [
            (true, true, 3),
            (true, false, 3),
            (false, true, 3),
            (false, false, 0),
            (true, true, 0),
        ] {
            let fixture = GqaFixture::new(q_lora, v_lora, rank);
            let (ordinary, calls) = observe_matmul_calls(|| fixture.ordinary());
            assert_eq!(
                calls,
                fixture.cache.seq_len * (4 + 2 * usize::from(q_lora) + 2 * usize::from(v_lora))
            );
            let dispatch = force_scalar();
            let (scalar, calls) = observe_matmul_calls(|| fixture.forced(&dispatch).unwrap());
            assert_eq!(calls, 0);
            assert_eq!(
                dispatch.verify_for(Operation::GqaBackward),
                Ok(Backend::Scalar)
            );
            for (i, (a, b)) in gqa_arrays(&ordinary)
                .into_iter()
                .zip(gqa_arrays(&scalar))
                .enumerate()
            {
                assert!(a.iter().chain(b).all(|v| v.is_finite()));
                if i == 4 || (rank > 0 && if i < 2 { q_lora } else { v_lora }) {
                    assert!(a.iter().any(|v| *v != 0.0), "array {i} lacks signal");
                }
                parity(a, b);
            }
        }
    }

    #[test]
    fn forced_gqa_zero_sparse_saturated_and_nonfinite_classifications() {
        for case in 0..5 {
            let mut fixture = GqaFixture::new(true, true, 3);
            match case {
                0 => fixture.dy.fill(0.0),
                1 => {
                    fixture.dy.fill(0.0);
                    fixture.dy[7] = 0.7;
                }
                2 => {
                    for (i, gate) in fixture.cache.gate_z.iter_mut().enumerate() {
                        *gate = if i % 2 == 0 { 80.0 } else { -80.0 };
                    }
                }
                3 => fixture.cache.q_h[1][0] = f32::NAN,
                _ => fixture.dy[7] = f32::INFINITY,
            }
            let ordinary = fixture.ordinary();
            let dispatch = force_scalar();
            let (scalar, calls) = observe_matmul_calls(|| fixture.forced(&dispatch).unwrap());
            assert_eq!(calls, 0);
            assert_eq!(
                dispatch.verify_for(Operation::GqaBackward),
                Ok(Backend::Scalar)
            );
            for (a, b) in gqa_arrays(&ordinary).into_iter().zip(gqa_arrays(&scalar)) {
                parity(a, b);
                if case == 0 {
                    assert!(b.iter().all(|v| *v == 0.0));
                }
            }
        }
    }

    #[test]
    fn forced_gqa_rejects_incomplete_shapes_before_execution() {
        let mutations: &[fn(&mut GqaFixture)] = &[
            |f| f.cache.seq_len = 0,
            |f| f.cache.num_kv_heads = 0,
            |f| f.cache.num_q_heads = 3,
            |f| f.cache.num_q_heads = usize::MAX - 1,
            |f| f.cache.rope_dim = 3,
            |f| {
                f.cache.x_input.pop();
            },
            |f| {
                f.cache.softmax_probs[2].pop();
            },
            |f| {
                f.cache.q_h[1].pop();
            },
            |f| {
                f.cache.cos_vals.pop();
            },
            |f| {
                f.cache.h_v.pop();
            },
            |f| {
                f.b_q.pop();
            },
            |f| {
                f.w_o.pop();
            },
            |f| {
                f.dy.pop();
            },
        ];
        for mutate in mutations {
            let mut fixture = GqaFixture::new(true, true, 3);
            mutate(&mut fixture);
            let dispatch = force_scalar();
            let (result, calls) = observe_matmul_calls(|| fixture.forced(&dispatch));
            assert!(matches!(
                result,
                Err(ForcedExecutionError::Shape {
                    operation: Operation::GqaBackward,
                    ..
                })
            ));
            assert_eq!(calls, 0);
            assert!(dispatch.verify().is_err());
        }
    }

    struct GdnFixture {
        saved: GdnSaved,
        weights: GatedDeltaNetWeights,
        dy: Vec<f32>,
    }

    impl GdnFixture {
        fn new(lora: bool) -> Self {
            Self::with_length(lora, 5)
        }

        fn with_length(lora: bool, t: usize) -> Self {
            let (hidden, kh, vh, k, v, kernel, rank) = (7, 1, 2, 3, 5, 3, 3);
            let qkv = 2 * kh * k + vh * v;
            let output = vh * v;
            let weights = GatedDeltaNetWeights {
                in_proj_qkv: values(qkv * hidden, 1),
                in_proj_qkv_rows: qkv,
                in_proj_qkv_cols: hidden,
                in_proj_z: values(output * hidden, 2),
                in_proj_z_rows: output,
                in_proj_z_cols: hidden,
                in_proj_a: values(vh * hidden, 3),
                in_proj_a_rows: vh,
                in_proj_a_cols: hidden,
                in_proj_b: values(vh * hidden, 4),
                in_proj_b_rows: vh,
                in_proj_b_cols: hidden,
                out_proj: values(hidden * output, 5),
                out_proj_rows: hidden,
                out_proj_cols: output,
                a_log: vec![-0.5; vh],
                dt_bias: vec![0.1; vh],
                norm_weight: vec![1.0; v],
                conv1d_weight: values(qkv * kernel, 6),
                conv_dim: qkv,
                kernel_size: kernel,
            };
            let mut saved = GdnSaved::new(
                t,
                kh,
                vh,
                k,
                v,
                hidden,
                qkv,
                output,
                kernel,
                1.0 / (k as f32).sqrt(),
                1e-6,
            );
            let a_qkv = values(rank * hidden, 7);
            let b_qkv = values(qkv * rank, 8);
            let a_z = values(rank * hidden, 9);
            let b_z = values(output * rank, 10);
            let a_b = values(rank * hidden, 11);
            let b_b = values(vh * rank, 12);
            let a_a = values(rank * hidden, 13);
            let b_a = values(vh * rank, 14);
            let a_out = values(rank * output, 15);
            let b_out = values(hidden * rank, 16);
            let cfg = Qwen35Config::qwen35_0_8b();
            gdn_forward_save(
                &values(t * hidden, 17),
                &weights,
                &cfg,
                &mut saved,
                &mut vec![0.0; t * hidden],
                lora.then_some(a_qkv.as_slice()),
                lora.then_some(b_qkv.as_slice()),
                lora.then_some(a_z.as_slice()),
                lora.then_some(b_z.as_slice()),
                lora.then_some(a_b.as_slice()),
                lora.then_some(b_b.as_slice()),
                lora.then_some(a_a.as_slice()),
                lora.then_some(b_a.as_slice()),
                lora.then_some(a_out.as_slice()),
                lora.then_some(b_out.as_slice()),
                rank,
                2.0,
            );
            Self {
                saved,
                weights,
                dy: values(t * hidden, 18),
            }
        }
    }

    #[test]
    fn forced_gdn_all_eleven_arrays_and_asymmetric_recurrence_bypass_gemm() {
        for lora in [false, true] {
            let fixture = GdnFixture::new(lora);
            let (ordinary, calls) = observe_matmul_calls(|| {
                automatic_gdn(&fixture.dy, &fixture.saved, &fixture.weights)
            });
            assert_eq!(calls, if lora { 10 * fixture.saved.seq_len } else { 0 });
            let dispatch = force_scalar();
            let (scalar, calls) = observe_matmul_calls(|| {
                dispatch
                    .gdn_backward(&fixture.dy, &fixture.saved, &fixture.weights)
                    .unwrap()
            });
            assert_eq!(calls, 0);
            assert_eq!(
                dispatch.verify_for(Operation::GdnBackward),
                Ok(Backend::Scalar)
            );
            for (i, (a, b)) in gdn_arrays(&ordinary)
                .into_iter()
                .zip(gdn_arrays(&scalar))
                .enumerate()
            {
                assert!(a.iter().chain(b).all(|v| v.is_finite()));
                if lora || i == 10 {
                    assert!(a.iter().any(|v| *v != 0.0), "array {i} lacks signal");
                } else {
                    assert!(a.is_empty());
                }
                parity(a, b);
            }
        }
    }

    #[test]
    fn forced_gdn_zero_sparse_saturated_and_nonfinite_classifications() {
        for case in 0..5 {
            let mut fixture = GdnFixture::new(true);
            match case {
                0 => fixture.dy.fill(0.0),
                1 => {
                    fixture.dy.fill(0.0);
                    fixture.dy[7] = 0.7;
                }
                2 => {
                    fixture.saved.beta_raw.fill(-80.0);
                    fixture.saved.beta.fill(0.0);
                    fixture.saved.alpha_proj.fill(80.0);
                }
                3 => fixture.dy[7] = f32::NAN,
                _ => fixture.dy[7] = f32::INFINITY,
            }
            let ordinary = automatic_gdn(&fixture.dy, &fixture.saved, &fixture.weights);
            let dispatch = force_scalar();
            let (scalar, calls) = observe_matmul_calls(|| {
                dispatch
                    .gdn_backward(&fixture.dy, &fixture.saved, &fixture.weights)
                    .unwrap()
            });
            assert_eq!(calls, 0);
            assert_eq!(
                dispatch.verify_for(Operation::GdnBackward),
                Ok(Backend::Scalar)
            );
            for (a, b) in gdn_arrays(&ordinary).into_iter().zip(gdn_arrays(&scalar)) {
                parity(a, b);
                if case == 0 {
                    assert!(b.iter().all(|v| *v == 0.0));
                }
            }
        }
    }

    #[test]
    fn forced_gdn_empty_sequence_preserves_empty_gradients() {
        for lora in [false, true] {
            let fixture = GdnFixture::with_length(lora, 0);
            let ordinary = automatic_gdn(&fixture.dy, &fixture.saved, &fixture.weights);
            let dispatch = force_scalar();
            let (scalar, calls) = observe_matmul_calls(|| {
                dispatch
                    .gdn_backward(&fixture.dy, &fixture.saved, &fixture.weights)
                    .unwrap()
            });
            assert_eq!(calls, 0);
            assert_eq!(
                dispatch.verify_for(Operation::GdnBackward),
                Ok(Backend::Scalar)
            );
            for (a, b) in gdn_arrays(&ordinary).into_iter().zip(gdn_arrays(&scalar)) {
                assert!(a.is_empty() && b.is_empty());
            }
        }
    }

    #[test]
    fn forced_gdn_rejects_incomplete_shapes_before_execution() {
        let mutations: &[fn(&mut GdnFixture)] = &[
            |f| f.saved.num_key_heads = 0,
            |f| f.saved.ratio = 3,
            |f| f.saved.key_dim = usize::MAX,
            |f| f.saved.kernel_size = 0,
            |f| f.saved.qkv_dim += 1,
            |f| {
                f.saved.s_after.pop();
            },
            |f| {
                f.saved.q_eps_norm.pop();
            },
            |f| {
                f.saved.conv_buffers.pop();
            },
            |f| {
                f.saved.h_a.clear();
            },
            |f| {
                f.saved.lora_b_out.pop();
            },
            |f| {
                f.weights.in_proj_a.pop();
            },
            |f| f.weights.out_proj_rows += 1,
            |f| {
                f.dy.pop();
            },
        ];
        for mutate in mutations {
            let mut fixture = GdnFixture::new(true);
            mutate(&mut fixture);
            let dispatch = force_scalar();
            let (result, calls) = observe_matmul_calls(|| {
                dispatch.gdn_backward(&fixture.dy, &fixture.saved, &fixture.weights)
            });
            assert!(matches!(
                result,
                Err(ForcedExecutionError::Shape {
                    operation: Operation::GdnBackward,
                    ..
                })
            ));
            assert_eq!(calls, 0);
            assert!(dispatch.verify().is_err());
        }
    }
}
