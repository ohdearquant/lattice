//! GPU-vs-CPU tiny-model tests and CPU reference helpers.
use super::*;
use crate::attention::softmax_row::{finalize_row, row_fails_closed_pre_exp, row_max_and_any_nan};

#[test]
fn tiny_model_matches_gpu_and_keeps_static_buffers() {
    let config = Qwen3Config {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 48,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let runtime = GpuRuntimeConfig {
        max_seq_len: 8,
        upload_embeddings_to_gpu: false,
    };
    let weights = make_test_weights(&config);
    let gpu = match GpuModelState::new(config.clone(), &weights, runtime) {
        Ok(gpu) => gpu,
        Err(GpuForwardError::NoAdapter) => return, // CI without GPU
        Err(e) => panic!("failed to init GPU state: {e}"),
    };

    let input_ids = vec![1, 7, 2, 31, 4];
    let before_buffers = gpu.user_buffer_creation_count();
    let before_submits = gpu.submit_count();

    let cpu = cpu_forward_reference(&weights, &config, &input_ids);
    let gpu_out = gpu
        .forward(&input_ids, input_ids.len())
        .expect("test setup: GPU forward should succeed");

    assert_eq!(before_buffers, gpu.user_buffer_creation_count());
    assert_eq!(before_submits + 1, gpu.submit_count());
    assert_eq!(cpu.len(), gpu_out.len());

    let mut max_abs = 0.0f32;
    for (a, b) in cpu.iter().zip(gpu_out.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(max_abs < 1e-3, "max abs diff = {max_abs}");
}

/// Same tolerance `tiny_model_matches_gpu_and_keeps_static_buffers` uses to
/// compare GPU output against its CPU reference.
const FAIL_CLOSED_MAX_ABS_DIFF: f32 = 1e-3;

/// ADR-080 C1 (#790) regression: `attention_softmax`'s WGSL kernel must fail
/// closed by ASSIGNMENT on a non-finite (NaN or +/-infinite) score, not
/// floor-clamp the denominator (`1.0 / max(sum_val, 1e-20)`) into a
/// finite-looking reciprocal that then multiplies an already-poisoned
/// numerator through.
///
/// Isolation: corrupt a single entry of one layer's `q_proj_weight` so that
/// exactly one attention head's Q vector is poisoned at every query
/// position, while K/V/embeddings/residual stay completely clean. Compares
/// the GPU output element-wise against a CPU reference built from the
/// crate's own fail-closed `softmax_row` helpers (not a plain `f32::max`-fold
/// reimplementation) at the same tolerance the sibling test above uses.
///
/// codex round-1 medium #2 on #795: the prior version of this test built the
/// 253-line CPU oracle (`cpu_forward_fail_closed_reference`) but only
/// compared NaN *counts* between GPU and CPU output, independently, never
/// the two arrays against each other -- so an arbitrary finite garbage value
/// in the poisoned row would have passed. `run_attention_softmax_fail_closed_case`
/// below asserts full element-wise agreement (this is the fix) and covers
/// all three non-finite classes the WGSL kernel's `is_non_finite` bitcast
/// check must catch: NaN, +infinity, and -infinity.
///
/// Mutation-sensitive both ways (reverified against this element-wise
/// oracle comparison, not just the old NaN-count check, in a disposable
/// worktree under the GPU lock):
/// - reverting `shaders.rs`'s `ATTENTION_SOFTMAX_SHADER` to the pre-fix
///   `1.0 / max(sum_scratch[0u], 1e-20)` floor-clamp (no pre-exp scan) makes
///   every case below fail (non-finite values leak into the GPU output while
///   the CPU reference stays all-finite).
/// - mutating either fail-closed branch's `SCORES[row_base + k] = 0.0;` to
///   write an arbitrary finite nonzero constant instead makes every case
///   below fail (the GPU row would still contain zero NaN/Inf elements, but
///   would now diverge from the CPU reference's true zero row, which only
///   the element-wise comparison this round added can detect).
fn run_attention_softmax_fail_closed_case(corrupt_value: f32) {
    let config = Qwen3Config {
        vocab_size: 8,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        intermediate_size: 48,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let runtime = GpuRuntimeConfig {
        max_seq_len: 8,
        upload_embeddings_to_gpu: false,
    };

    let mut weights = make_test_weights(&config);
    let hidden = config.hidden_size;
    let corrupt_head: usize = 1;
    let corrupt_head_dim: usize = 3;
    let corrupt_out_idx = corrupt_head * config.head_dim + corrupt_head_dim;
    let corrupt_in_idx = 5usize;
    weights.layers[0].q_proj_weight[corrupt_out_idx * hidden + corrupt_in_idx] = corrupt_value;

    let gpu = match GpuModelState::new(config.clone(), &weights, runtime) {
        Ok(gpu) => gpu,
        Err(GpuForwardError::NoAdapter) => return, // CI without GPU
        Err(e) => panic!("failed to init GPU state: {e}"),
    };

    let input_ids: Vec<u32> = vec![1, 3, 2, 5, 6];
    let gpu_out = gpu
        .forward(&input_ids, input_ids.len())
        .expect("GPU forward should not itself error");

    let cpu_ref = cpu_forward_fail_closed_reference(&weights, &config, &input_ids);
    assert_eq!(gpu_out.len(), cpu_ref.len());

    let non_finite_ref = cpu_ref.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite_ref, 0,
        "test bug: the fail-closed CPU reference itself produced a \
         non-finite value (corrupt_value = {corrupt_value})"
    );

    let non_finite_gpu = gpu_out.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite_gpu,
        0,
        "attention_softmax must fail closed (zero row) on a \
         {corrupt_value}-poisoned Q lane instead of floor-clamping the \
         denominator (ADR-080 C1, #790); got {non_finite_gpu}/{} \
         non-finite elements",
        gpu_out.len()
    );

    let mut max_abs = 0.0f32;
    for (a, b) in cpu_ref.iter().zip(gpu_out.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < FAIL_CLOSED_MAX_ABS_DIFF,
        "GPU output diverges element-wise from the fail-closed CPU \
         reference (corrupt_value = {corrupt_value}): max abs diff = \
         {max_abs} (tolerance {FAIL_CLOSED_MAX_ABS_DIFF})"
    );
}

#[test]
fn attention_softmax_fails_closed_on_nan_q_lane() {
    run_attention_softmax_fail_closed_case(f32::NAN);
}

#[test]
fn attention_softmax_fails_closed_on_positive_infinity_q_lane() {
    run_attention_softmax_fail_closed_case(f32::INFINITY);
}

#[test]
fn attention_softmax_fails_closed_on_negative_infinity_q_lane() {
    run_attention_softmax_fail_closed_case(f32::NEG_INFINITY);
}

/// Same tiny-model math as [`cpu_forward_reference`], but the softmax step
/// applies the crate's own ADR-080 C1 contract helpers explicitly
/// (`softmax_row`) instead of a plain `f32::max` fold — this reference is
/// provably contract-compliant rather than "whatever the naive scalar loop
/// happens to do with a NaN input".
fn cpu_forward_fail_closed_reference(
    weights: &Qwen3Weights,
    config: &Qwen3Config,
    input_ids: &[u32],
) -> Vec<f32> {
    let seq_len = input_ids.len();
    let hidden = config.hidden_size;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let head_dim = config.head_dim;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let groups = config.num_groups();
    let intermediate = config.intermediate_size;

    let (rope_cos, rope_sin) = build_rope_tables(head_dim, seq_len, config.rope_theta);

    let mut hidden_buf = vec![0.0f32; seq_len * hidden];
    for (i, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        hidden_buf[i * hidden..(i + 1) * hidden]
            .copy_from_slice(&weights.embed_tokens[tok * hidden..(tok + 1) * hidden]);
    }

    let mut residual = vec![0.0f32; seq_len * hidden];
    let mut q = vec![0.0f32; seq_len * q_dim];
    let mut k = vec![0.0f32; seq_len * kv_dim];
    let mut v = vec![0.0f32; seq_len * kv_dim];
    let mut attn_out = vec![0.0f32; seq_len * q_dim];
    let mut gate = vec![0.0f32; seq_len * intermediate];
    let mut up = vec![0.0f32; seq_len * intermediate];

    for layer in &weights.layers {
        residual.copy_from_slice(&hidden_buf);
        rms_norm_cpu(
            &mut hidden_buf,
            &layer.input_layernorm_weight,
            hidden,
            config.rms_norm_eps,
        );

        matmul_bt_cpu(
            &hidden_buf,
            &layer.q_proj_weight,
            &mut q,
            seq_len,
            hidden,
            q_dim,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.k_proj_weight,
            &mut k,
            seq_len,
            hidden,
            kv_dim,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.v_proj_weight,
            &mut v,
            seq_len,
            hidden,
            kv_dim,
        );

        rms_norm_cpu(&mut q, &layer.q_norm_weight, head_dim, config.rms_norm_eps);
        rms_norm_cpu(&mut k, &layer.k_norm_weight, head_dim, config.rms_norm_eps);
        apply_rope_cpu(&mut q, seq_len, num_heads, head_dim, &rope_cos, &rope_sin);
        apply_rope_cpu(
            &mut k,
            seq_len,
            num_kv_heads,
            head_dim,
            &rope_cos,
            &rope_sin,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let kv_h = h / groups;
            for qi in 0..seq_len {
                let n_valid = qi + 1;
                let mut row = vec![0.0f32; n_valid];
                for (ki, slot) in row.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = (qi * num_heads + h) * head_dim + d;
                        let k_idx = (ki * num_kv_heads + kv_h) * head_dim + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    *slot = dot * scale;
                }

                // --- ADR-080 C1 contract, applied via the crate's own shared
                // helper functions (not a reimplementation): ---
                let (max_val, any_nan) = row_max_and_any_nan(&row);
                if row_fails_closed_pre_exp(max_val, any_nan) {
                    row.fill(0.0);
                } else {
                    let mut sum = 0.0f32;
                    for x in row.iter_mut() {
                        let e = (*x - max_val).exp();
                        *x = e;
                        sum += e;
                    }
                    finalize_row(&mut row, sum);
                }

                for h_ in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (ki, &p) in row.iter().enumerate() {
                        let v_idx = (ki * num_kv_heads + kv_h) * head_dim + h_;
                        acc += p * v[v_idx];
                    }
                    attn_out[(qi * num_heads + h) * head_dim + h_] = acc;
                }
            }
        }

        matmul_bt_cpu(
            &attn_out,
            &layer.o_proj_weight,
            &mut hidden_buf,
            seq_len,
            q_dim,
            hidden,
        );
        add_inplace_cpu(&mut hidden_buf, &residual);

        residual.copy_from_slice(&hidden_buf);
        rms_norm_cpu(
            &mut hidden_buf,
            &layer.post_attention_layernorm_weight,
            hidden,
            config.rms_norm_eps,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.gate_proj_weight,
            &mut gate,
            seq_len,
            hidden,
            intermediate,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.up_proj_weight,
            &mut up,
            seq_len,
            hidden,
            intermediate,
        );
        silu_inplace_cpu(&mut gate);
        mul_inplace_cpu(&mut gate, &up);
        matmul_bt_cpu(
            &gate,
            &layer.down_proj_weight,
            &mut hidden_buf,
            seq_len,
            intermediate,
            hidden,
        );
        add_inplace_cpu(&mut hidden_buf, &residual);
    }

    rms_norm_cpu(
        &mut hidden_buf,
        &weights.norm_weight,
        hidden,
        config.rms_norm_eps,
    );
    hidden_buf
}

fn make_test_weights(config: &Qwen3Config) -> Qwen3Weights {
    let mut rng = Lcg::new(0x5EED_BAAD_F00Du64);
    let hidden = config.hidden_size;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let intermediate = config.intermediate_size;

    let embed_tokens = random_vec(&mut rng, config.vocab_size * hidden);
    let norm_weight = random_positive_vec(&mut rng, hidden);
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for _ in 0..config.num_hidden_layers {
        layers.push(Qwen3LayerWeights {
            q_proj_weight: random_vec(&mut rng, q_dim * hidden),
            k_proj_weight: random_vec(&mut rng, kv_dim * hidden),
            v_proj_weight: random_vec(&mut rng, kv_dim * hidden),
            o_proj_weight: random_vec(&mut rng, hidden * q_dim),
            q_norm_weight: random_positive_vec(&mut rng, config.head_dim),
            k_norm_weight: random_positive_vec(&mut rng, config.head_dim),
            input_layernorm_weight: random_positive_vec(&mut rng, hidden),
            gate_proj_weight: random_vec(&mut rng, intermediate * hidden),
            up_proj_weight: random_vec(&mut rng, intermediate * hidden),
            down_proj_weight: random_vec(&mut rng, hidden * intermediate),
            post_attention_layernorm_weight: random_positive_vec(&mut rng, hidden),
        });
    }
    Qwen3Weights {
        embed_tokens,
        layers,
        norm_weight,
    }
}

fn cpu_forward_reference(
    weights: &Qwen3Weights,
    config: &Qwen3Config,
    input_ids: &[u32],
) -> Vec<f32> {
    let seq_len = input_ids.len();
    let hidden = config.hidden_size;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let head_dim = config.head_dim;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let groups = config.num_groups();
    let intermediate = config.intermediate_size;

    let (rope_cos, rope_sin) = build_rope_tables(head_dim, seq_len, config.rope_theta);

    let mut hidden_buf = vec![0.0f32; seq_len * hidden];
    for (i, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        hidden_buf[i * hidden..(i + 1) * hidden]
            .copy_from_slice(&weights.embed_tokens[tok * hidden..(tok + 1) * hidden]);
    }

    let mut residual = vec![0.0f32; seq_len * hidden];
    let mut q = vec![0.0f32; seq_len * q_dim];
    let mut k = vec![0.0f32; seq_len * kv_dim];
    let mut v = vec![0.0f32; seq_len * kv_dim];
    let mut scores = vec![0.0f32; num_heads * seq_len * seq_len];
    let mut attn_out = vec![0.0f32; seq_len * q_dim];
    let mut gate = vec![0.0f32; seq_len * intermediate];
    let mut up = vec![0.0f32; seq_len * intermediate];

    for layer in &weights.layers {
        residual.copy_from_slice(&hidden_buf);
        rms_norm_cpu(
            &mut hidden_buf,
            &layer.input_layernorm_weight,
            hidden,
            config.rms_norm_eps,
        );

        matmul_bt_cpu(
            &hidden_buf,
            &layer.q_proj_weight,
            &mut q,
            seq_len,
            hidden,
            q_dim,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.k_proj_weight,
            &mut k,
            seq_len,
            hidden,
            kv_dim,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.v_proj_weight,
            &mut v,
            seq_len,
            hidden,
            kv_dim,
        );

        rms_norm_cpu(&mut q, &layer.q_norm_weight, head_dim, config.rms_norm_eps);
        rms_norm_cpu(&mut k, &layer.k_norm_weight, head_dim, config.rms_norm_eps);
        apply_rope_cpu(&mut q, seq_len, num_heads, head_dim, &rope_cos, &rope_sin);
        apply_rope_cpu(
            &mut k,
            seq_len,
            num_kv_heads,
            head_dim,
            &rope_cos,
            &rope_sin,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let kv_h = h / groups;
            for qi in 0..seq_len {
                let row_base = (h * seq_len + qi) * seq_len;
                let mut max_val = f32::NEG_INFINITY;
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = (qi * num_heads + h) * head_dim + d;
                        let k_idx = (ki * num_kv_heads + kv_h) * head_dim + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    let val = if ki <= qi {
                        dot * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    scores[row_base + ki] = val;
                    max_val = max_val.max(val);
                }
                let mut sum = 0.0f32;
                for ki in 0..seq_len {
                    let x = if ki <= qi {
                        let e = (scores[row_base + ki] - max_val).exp();
                        scores[row_base + ki] = e;
                        e
                    } else {
                        scores[row_base + ki] = 0.0;
                        0.0
                    };
                    sum += x;
                }
                let inv_sum = 1.0 / sum.max(1e-20);
                for ki in 0..seq_len {
                    scores[row_base + ki] *= inv_sum;
                }
            }
        }

        for qi in 0..seq_len {
            for h in 0..num_heads {
                let kv_h = h / groups;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for ki in 0..seq_len {
                        let p = scores[(h * seq_len + qi) * seq_len + ki];
                        let v_idx = (ki * num_kv_heads + kv_h) * head_dim + d;
                        acc += p * v[v_idx];
                    }
                    attn_out[(qi * num_heads + h) * head_dim + d] = acc;
                }
            }
        }

        matmul_bt_cpu(
            &attn_out,
            &layer.o_proj_weight,
            &mut hidden_buf,
            seq_len,
            q_dim,
            hidden,
        );
        add_inplace_cpu(&mut hidden_buf, &residual);

        residual.copy_from_slice(&hidden_buf);
        rms_norm_cpu(
            &mut hidden_buf,
            &layer.post_attention_layernorm_weight,
            hidden,
            config.rms_norm_eps,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.gate_proj_weight,
            &mut gate,
            seq_len,
            hidden,
            intermediate,
        );
        matmul_bt_cpu(
            &hidden_buf,
            &layer.up_proj_weight,
            &mut up,
            seq_len,
            hidden,
            intermediate,
        );
        silu_inplace_cpu(&mut gate);
        mul_inplace_cpu(&mut gate, &up);
        matmul_bt_cpu(
            &gate,
            &layer.down_proj_weight,
            &mut hidden_buf,
            seq_len,
            intermediate,
            hidden,
        );
        add_inplace_cpu(&mut hidden_buf, &residual);
    }

    rms_norm_cpu(
        &mut hidden_buf,
        &weights.norm_weight,
        hidden,
        config.rms_norm_eps,
    );
    hidden_buf
}

fn matmul_bt_cpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = acc;
        }
    }
}

fn rms_norm_cpu(x: &mut [f32], gamma: &[f32], row_len: usize, eps: f32) {
    let rows = x.len() / row_len;
    for r in 0..rows {
        let row = &mut x[r * row_len..(r + 1) * row_len];
        let mut sum_sq = 0.0f32;
        for &v in row.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / row_len as f32 + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(gamma.iter()) {
            *v = *v * inv_rms * g;
        }
    }
}

fn apply_rope_cpu(
    x: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for pos in 0..seq_len {
        for h in 0..num_heads {
            let base = (pos * num_heads + h) * head_dim;
            let rope_base = pos * half;
            for i in 0..half {
                let x0 = x[base + i];
                let x1 = x[base + half + i];
                let c = cos[rope_base + i];
                let s = sin[rope_base + i];
                x[base + i] = x0 * c - x1 * s;
                x[base + half + i] = x0 * s + x1 * c;
            }
        }
    }
}

fn silu_inplace_cpu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * (1.0 / (1.0 + (-*v).exp()));
    }
}

fn mul_inplace_cpu(a: &mut [f32], b: &[f32]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av *= bv;
    }
}

fn add_inplace_cpu(a: &mut [f32], b: &[f32]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av += bv;
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        let x = self.next_u32() as f32 / u32::MAX as f32;
        (x - 0.5) * 0.2
    }
}

fn random_vec(rng: &mut Lcg, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.next_f32()).collect()
}

fn random_positive_vec(rng: &mut Lcg, len: usize) -> Vec<f32> {
    (0..len).map(|_| 0.5 + rng.next_f32().abs()).collect()
}
