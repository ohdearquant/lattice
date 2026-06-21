//! Forward-parity differential tests for the LoRA backward surface.
//!
//! These tests MEASURE divergence between the materialised forward used
//! by the backward tape and the real Qwen3.5 model primitives. They do
//! NOT modify production code. The numbers reported here are the gate for
//! any on-par claim (Leo's bar: forward must match the real model under
//! 1e-3 before an on-par claim is valid).
//!
//! Run:
//!   cargo test -p lattice-inference --test lora_forward_parity_test \
//!       --features train-backward -- --nocapture
//!
//! TEST 1: RMSNorm convention diff
//!   tape.rs uses plain gamma: xi * wi * inv_rms
//!   qwen35_rms_norm uses shifted gamma: xi * inv_rms * (1 + gi)
//!   Report: elementwise max-diff between the two on the same inputs.
//!
//! TEST 2: GQA single-layer forward parity, materialised vs oracle
//!   Path A: gqa_forward_with_cache (materialised backward forward)
//!   Path B: oracle built from the REAL primitives (replicated inline):
//!     - real_rms_norm_shifted (matches qwen35_rms_norm exactly, shifted gamma)
//!     - deinterleave_q_gate + apply_sigmoid_gate (same functions from crate)
//!     - same RoPE (stride-half convention, matches real model)
//!     - same attention/o_proj arithmetic
//!   This exposes convention bugs in the materialised forward.
//!
//! TEST 3: Hypothetical shifted-gamma tape variant
//!   What max-diff WOULD be if tape.rs rms_norm_forward used shifted gamma,
//!   computed inline in the test without modifying production code.

#[cfg(feature = "train-backward")]
mod parity_tests {
    use lattice_inference::attention::gated::{apply_sigmoid_gate, deinterleave_q_gate};
    use lattice_inference::backward::attention_gqa::gqa_forward_with_cache;
    use lattice_inference::backward::tape::rms_norm_forward;

    // ===================================================================
    // Inline oracle: exact copy of qwen35_rms_norm (pub(crate) so not
    // importable from integration tests — but replicating 10 lines is
    // correct and avoids a visibility-change to production code).
    //
    // Source: crates/inference/src/model/qwen35/norm.rs:5-22
    // Convention: v * inv_rms * (1 + g)  (shifted gamma)
    // ===================================================================

    fn real_rms_norm_shifted(x: &mut [f32], gamma: &[f32], hidden: usize, eps: f32) {
        let num_tokens = x.len() / hidden;
        for t in 0..num_tokens {
            let row = &mut x[t * hidden..(t + 1) * hidden];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            for (v, &g) in row.iter_mut().zip(gamma.iter()) {
                *v = *v * inv_rms * (1.0 + g);
            }
        }
    }

    // ===================================================================
    // Utilities
    // ===================================================================

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "max_abs_diff: length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// xorshift64 PRNG matching existing test conventions in this codebase.
    fn rand_vec(state: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            out.push(((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    // ===================================================================
    // TEST 1: RMSNorm convention diff
    //
    // tape.rs rms_norm_forward:  normed[i] = xi * wi * inv_rms
    //   (plain gamma: w is multiplied directly)
    //
    // qwen35_rms_norm (real model): normed[i] = xi * inv_rms * (1 + gi)
    //   (shifted gamma: adds 1 to gamma before multiplying)
    //
    // Theoretical diff = real - tape = xi * inv_rms * (1+gi) - xi*gi*inv_rms
    //                  = xi * inv_rms   (the "+1" term, independent of gamma)
    // ===================================================================

    #[test]
    fn test1_rms_norm_convention_diff() {
        let hidden = 64usize;
        let eps = 1e-6f32;
        let mut rng = 0xCAFE_BABE_u64;

        // Nonzero weights (important: w=0 would collapse to the same when
        // gamma_shifted=0 → (1+0)=1 → same as plain gamma with w=1, so we
        // need w != 0 and w != 1 to clearly see the convention divergence)
        let x = rand_vec(&mut rng, hidden, 1.0);
        let w = rand_vec(&mut rng, hidden, 0.5); // gamma values in [-0.5, 0.5]

        // Path A: tape.rs rms_norm_forward — plain gamma xi*wi*inv_rms
        let (tape_out, tape_inv_rms) = rms_norm_forward(&x, &w, eps);

        // Path B: real shifted gamma xi*(1+wi)*inv_rms
        let mut real_out = x.clone();
        real_rms_norm_shifted(&mut real_out, &w, hidden, eps);

        let diff = max_abs_diff(&tape_out, &real_out);

        let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / hidden as f32;
        let expected_inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // Analytical diff = xi * inv_rms (the constant +1 term)
        let theoretical_diff: Vec<f32> = x.iter().map(|xi| xi * tape_inv_rms).collect();
        let actual_diff: Vec<f32> = real_out
            .iter()
            .zip(tape_out.iter())
            .map(|(r, t)| r - t)
            .collect();
        let meta_diff = max_abs_diff(&theoretical_diff, &actual_diff);

        println!("=== TEST 1: RMSNorm convention diff ===");
        println!("  hidden={hidden}, eps={eps}");
        println!("  inv_rms (tape): {tape_inv_rms:.8}");
        println!("  inv_rms (expected): {expected_inv_rms:.8}");
        println!("  Sample w[0..4]: {:?}", &w[..4]);
        println!("  Sample x[0..4]: {:?}", &x[..4]);
        println!("  tape_out[0..4]:  {:?}", &tape_out[..4]);
        println!("  real_out[0..4]:  {:?}", &real_out[..4]);
        println!("  MEASURED max-diff(tape vs real): {diff:.6e}");
        println!("  Theoretical diff = xi*inv_rms (the +1 term):");
        println!("    theoretical_diff[0..4]: {:?}", &theoretical_diff[..4]);
        println!("    actual_diff[0..4]:      {:?}", &actual_diff[..4]);
        println!("    max-diff(theoretical vs actual diff): {meta_diff:.6e}");
        println!();
    }

    // ===================================================================
    // TEST 2: GQA single-layer forward parity — materialised vs oracle
    //
    // Structural fixture mirrors Qwen3.5-0.8B layer 23 (the last full-
    // attention layer). We use small dims to keep the test fast:
    //   hidden=64, num_q_heads=4, num_kv_heads=2, head_dim=16, rank=8
    //
    // Both paths receive the same weights, same LoRA, same input x.
    //
    // PATH A: gqa_forward_with_cache  (materialised backward forward)
    //   This function uses SHIFTED q_norm/k_norm (1+gamma) as of the
    //   fix in attention_gqa.rs:445-466. So the q/k-norm convention is
    //   already correct. Any remaining divergence comes from other places.
    //
    // PATH B: Oracle from real primitives (NOT driving the loaded model)
    //   - real_rms_norm_shifted (exact qwen35_rms_norm replica) for q/k norm
    //   - same deinterleave_q_gate, apply_sigmoid_gate from crate
    //   - same stride-half RoPE
    //   - same causal softmax attention (seq_len=1 → trivial: single pos)
    //   - same o_proj matmul
    //
    // The oracle uses seq_len=1 for simplicity (position 0 only). The
    // real model's decode step also processes one token at a time.
    // ===================================================================

    #[test]
    fn test2_gqa_layer_forward_parity_materialised_vs_oracle() {
        let seq_len = 1usize;
        let hidden = 64usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 16usize;
        let rope_dim = 16usize; // full rope on all head dims
        let lora_rank = 8usize;
        let lora_scale = 1.0f32;

        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let half = rope_dim / 2;
        let eps = 1e-6f32;

        let mut rng = 0xDEAD_C0DE_u64;

        let w_q = rand_vec(&mut rng, 2 * q_dim * hidden, 0.1);
        let w_k = rand_vec(&mut rng, kv_dim * hidden, 0.1);
        let w_v = rand_vec(&mut rng, kv_dim * hidden, 0.1);
        let w_o = rand_vec(&mut rng, hidden * q_dim, 0.1);
        let q_norm_w = rand_vec(&mut rng, head_dim, 0.3); // nonzero gamma
        let k_norm_w = rand_vec(&mut rng, head_dim, 0.3);
        let x = rand_vec(&mut rng, seq_len * hidden, 0.5);

        // LoRA (nonzero B so delta is nontrivial)
        let lora_a_q = rand_vec(&mut rng, lora_rank * hidden, 0.05);
        let lora_b_q = rand_vec(&mut rng, 2 * q_dim * lora_rank, 0.05);
        let lora_a_v = rand_vec(&mut rng, lora_rank * hidden, 0.05);
        let lora_b_v = rand_vec(&mut rng, kv_dim * lora_rank, 0.05);

        // RoPE tables (position=0)
        let cos_table: Vec<f32> = (0..seq_len * half)
            .map(|i| {
                let pos = i / half;
                let dim_i = i % half;
                let theta = (pos as f32) / 10000f32.powf(2.0 * dim_i as f32 / rope_dim as f32);
                theta.cos()
            })
            .collect();
        let sin_table: Vec<f32> = (0..seq_len * half)
            .map(|i| {
                let pos = i / half;
                let dim_i = i % half;
                let theta = (pos as f32) / 10000f32.powf(2.0 * dim_i as f32 / rope_dim as f32);
                theta.sin()
            })
            .collect();

        // ---------------------------------------------------------------
        // PATH A: materialised forward
        // ---------------------------------------------------------------
        let (path_a_out, _cache_a) = gqa_forward_with_cache(
            &x,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &q_norm_w,
            &k_norm_w,
            Some(&lora_a_q),
            Some(&lora_b_q),
            Some(&lora_a_v),
            Some(&lora_b_v),
            lora_rank,
            lora_scale,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            &cos_table,
            &sin_table,
            eps,
        );

        // ---------------------------------------------------------------
        // PATH B: Oracle from real Qwen3.5 primitives (replicated inline)
        //
        // Mirrors full_attention_step_from_attn_out + project_qkv from
        // crates/inference/src/model/qwen35/forward.rs:170-246
        // ---------------------------------------------------------------
        let _scale_attn = 1.0f32 / (head_dim as f32).sqrt();
        let groups = num_q_heads / num_kv_heads;

        let mut oracle_out = vec![0.0f32; seq_len * hidden];

        // Capture intermediates for localisation
        let mut oracle_q_normed = vec![0.0f32; q_dim];
        let mut oracle_k_normed = vec![0.0f32; kv_dim];

        for t in 0..seq_len {
            let x_t = &x[t * hidden..(t + 1) * hidden];

            // 1. q+gate projection: w_q [2*q_dim, hidden], row-major
            let mut q_and_gate = vec![0.0f32; 2 * q_dim];
            for i in 0..2 * q_dim {
                let row = &w_q[i * hidden..(i + 1) * hidden];
                q_and_gate[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
            }
            // LoRA Q delta
            {
                let mut h_q = vec![0.0f32; lora_rank];
                for r in 0..lora_rank {
                    h_q[r] = lora_a_q[r * hidden..(r + 1) * hidden]
                        .iter()
                        .zip(x_t.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }
                for i in 0..2 * q_dim {
                    let acc: f32 = lora_scale
                        * lora_b_q[i * lora_rank..(i + 1) * lora_rank]
                            .iter()
                            .zip(h_q.iter())
                            .map(|(b, hi)| b * hi)
                            .sum::<f32>();
                    q_and_gate[i] += acc;
                }
            }

            // 2. deinterleave: per-head [Q_h | gate_h] layout
            let mut q_buf = vec![0.0f32; q_dim];
            let mut gate_z = vec![0.0f32; q_dim];
            deinterleave_q_gate(&q_and_gate, &mut q_buf, &mut gate_z, num_q_heads, head_dim);

            // 3. k_proj
            let mut k_buf = vec![0.0f32; kv_dim];
            for i in 0..kv_dim {
                let row = &w_k[i * hidden..(i + 1) * hidden];
                k_buf[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
            }

            // 4. v_proj + LoRA V
            let mut v_buf = vec![0.0f32; kv_dim];
            for i in 0..kv_dim {
                let row = &w_v[i * hidden..(i + 1) * hidden];
                v_buf[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
            }
            {
                let mut h_v = vec![0.0f32; lora_rank];
                for r in 0..lora_rank {
                    h_v[r] = lora_a_v[r * hidden..(r + 1) * hidden]
                        .iter()
                        .zip(x_t.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }
                for i in 0..kv_dim {
                    let acc: f32 = lora_scale
                        * lora_b_v[i * lora_rank..(i + 1) * lora_rank]
                            .iter()
                            .zip(h_v.iter())
                            .map(|(b, hi)| b * hi)
                            .sum::<f32>();
                    v_buf[i] += acc;
                }
            }

            // 5. q_norm: real shifted gamma per head
            for qh in 0..num_q_heads {
                let start = qh * head_dim;
                real_rms_norm_shifted(
                    &mut q_buf[start..start + head_dim],
                    &q_norm_w,
                    head_dim,
                    eps,
                );
            }

            // 6. k_norm: real shifted gamma per head
            for kvh in 0..num_kv_heads {
                let start = kvh * head_dim;
                real_rms_norm_shifted(
                    &mut k_buf[start..start + head_dim],
                    &k_norm_w,
                    head_dim,
                    eps,
                );
            }

            // Capture for localisation
            if t == 0 {
                oracle_q_normed.copy_from_slice(&q_buf);
                oracle_k_normed.copy_from_slice(&k_buf);
            }

            // 7. RoPE stride-half (matches gqa_forward_with_cache:481-506)
            let cos_t = &cos_table[t * half..(t + 1) * half];
            let sin_t = &sin_table[t * half..(t + 1) * half];
            for qh in 0..num_q_heads {
                let start = qh * head_dim;
                for i in 0..half {
                    let c = cos_t[i];
                    let s = sin_t[i];
                    let x0 = q_buf[start + i];
                    let x1 = q_buf[start + half + i];
                    q_buf[start + i] = x0 * c - x1 * s;
                    q_buf[start + half + i] = x0 * s + x1 * c;
                }
            }
            for kvh in 0..num_kv_heads {
                let start = kvh * head_dim;
                let x0_vals: Vec<f32> = (0..half).map(|i| k_buf[start + i]).collect();
                let x1_vals: Vec<f32> = (0..half).map(|i| k_buf[start + half + i]).collect();
                for i in 0..half {
                    let c = cos_t[i];
                    let s = sin_t[i];
                    k_buf[start + i] = x0_vals[i] * c - x1_vals[i] * s;
                    k_buf[start + half + i] = x0_vals[i] * s + x1_vals[i] * c;
                }
            }

            // 8. Causal attention (seq_len=1: position 0 can only attend to 0)
            //    Softmax over 1 element = 1.0, context = v[0]
            let mut context = vec![0.0f32; q_dim];
            for qh in 0..num_q_heads {
                let kvh = qh / groups;
                let q_off = qh * head_dim;
                let kv_off = kvh * head_dim;
                let q = &q_buf[q_off..q_off + head_dim];
                let k = &k_buf[kv_off..kv_off + head_dim];
                let v = &v_buf[kv_off..kv_off + head_dim];

                // Single-element softmax: attention score = exp(dot*scale)/exp(dot*scale) = 1.0
                // so context = 1.0 * v regardless of the score value
                let _dot: f32 = q.iter().zip(k.iter()).map(|(qi, ki)| qi * ki).sum();
                let ctx_off = qh * head_dim;
                context[ctx_off..ctx_off + head_dim].copy_from_slice(&v[..head_dim]);
            }

            // 9. Sigmoid gate: context[i] *= sigmoid(gate_z[i])
            apply_sigmoid_gate(&mut context, &gate_z);

            // 10. o_proj: w_o [hidden, q_dim]
            let out_t = &mut oracle_out[t * hidden..(t + 1) * hidden];
            for i in 0..hidden {
                let row = &w_o[i * q_dim..(i + 1) * q_dim];
                out_t[i] = row.iter().zip(context.iter()).map(|(a, b)| a * b).sum();
            }
        }

        // ---------------------------------------------------------------
        // Compare overall
        // ---------------------------------------------------------------
        let max_diff_overall = max_abs_diff(&path_a_out, &oracle_out);

        // ---------------------------------------------------------------
        // Localise: capture materialised q_norm for comparison
        // Re-derive the materialised q_norm output from the same weights.
        // attention_gqa.rs:448-456 uses the SHIFTED form (1+gamma)*inv_rms
        // ---------------------------------------------------------------
        let x_t = &x[..hidden];
        let mut q_and_gate_mat = vec![0.0f32; 2 * q_dim];
        for i in 0..2 * q_dim {
            let row = &w_q[i * hidden..(i + 1) * hidden];
            q_and_gate_mat[i] = row.iter().zip(x_t.iter()).map(|(a, b)| a * b).sum();
        }
        {
            let mut h_q = vec![0.0f32; lora_rank];
            for r in 0..lora_rank {
                h_q[r] = lora_a_q[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x_t.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
            for i in 0..2 * q_dim {
                let acc: f32 = lora_scale
                    * lora_b_q[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h_q.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                q_and_gate_mat[i] += acc;
            }
        }
        let mut q_mat_raw = vec![0.0f32; q_dim];
        let mut gate_z_mat = vec![0.0f32; q_dim];
        deinterleave_q_gate(
            &q_and_gate_mat,
            &mut q_mat_raw,
            &mut gate_z_mat,
            num_q_heads,
            head_dim,
        );

        // Materialised q_norm: SHIFTED (1+gamma)*inv_rms per head
        // (matches attention_gqa.rs:448-456 which was fixed to use shifted gamma)
        let mut q_mat_normed = q_mat_raw.clone();
        for qh in 0..num_q_heads {
            let start = qh * head_dim;
            let q_head = &mut q_mat_normed[start..start + head_dim];
            let mean_sq: f32 = q_head.iter().map(|xi| xi * xi).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for (j, qj) in q_head.iter_mut().enumerate() {
                *qj *= (1.0 + q_norm_w[j]) * inv_rms; // SHIFTED — matches attention_gqa.rs
            }
        }

        let qnorm_diff = max_abs_diff(&q_mat_normed, &oracle_q_normed);

        // Pre-attn RMSNorm check (tape.rs rms_norm_forward vs real)
        // This is what tape.rs uses for the pre-attention norm in the full training path.
        let w_all_ones = vec![1.0f32; hidden];
        let (tape_normed_ones, _) = rms_norm_forward(x_t, &w_all_ones, eps);
        let mut real_normed_ones = x_t.to_vec();
        real_rms_norm_shifted(&mut real_normed_ones, &w_all_ones, hidden, eps);
        let pre_attn_norm_diff_ones = max_abs_diff(&tape_normed_ones, &real_normed_ones);

        // With nonzero gamma
        let mut rng2 = 0xABCD_1234_u64;
        let w_nonzero = rand_vec(&mut rng2, hidden, 0.3);
        let (tape_normed_nonzero, _) = rms_norm_forward(x_t, &w_nonzero, eps);
        let mut real_normed_nonzero = x_t.to_vec();
        real_rms_norm_shifted(&mut real_normed_nonzero, &w_nonzero, hidden, eps);
        let pre_attn_norm_diff_nonzero = max_abs_diff(&tape_normed_nonzero, &real_normed_nonzero);

        println!("=== TEST 2: GQA layer forward parity (materialised vs oracle) ===");
        println!("  Approach: Path B = Oracle from real Qwen3.5 primitives (replicated inline)");
        println!(
            "  Dims: seq_len={seq_len} hidden={hidden} q_heads={num_q_heads} kv_heads={num_kv_heads}"
        );
        println!("        head_dim={head_dim} rope_dim={rope_dim} lora_rank={lora_rank}");
        println!("  path_a_out [0..4]: {:?}", &path_a_out[..4.min(hidden)]);
        println!("  oracle_out [0..4]: {:?}", &oracle_out[..4.min(hidden)]);
        println!();
        println!("  --- Overall ---");
        println!("  MEASURED max-diff(materialised vs oracle): {max_diff_overall:.6e}");
        println!();
        println!("  --- Localisation breakdown ---");
        println!("  post q_norm max-diff (materialised vs oracle): {qnorm_diff:.6e}");
        println!(
            "  pre-attn RMSNorm diff (tape vs real, w=all-ones): {pre_attn_norm_diff_ones:.6e}"
        );
        println!(
            "  pre-attn RMSNorm diff (tape vs real, w=nonzero):  {pre_attn_norm_diff_nonzero:.6e}"
        );
        println!();
        println!("  Interpretation:");
        if max_diff_overall < 1e-5 {
            println!("  [PASS] Overall diff < 1e-5: materialised forward matches oracle.");
        } else if max_diff_overall < 1e-3 {
            println!(
                "  [WARN] Overall diff in (1e-5, 1e-3): small divergence, not on-par threshold."
            );
        } else {
            println!("  [FAIL] Overall diff >= 1e-3: materialised forward does NOT match oracle.");
        }
    }

    // ===================================================================
    // TEST 3: Hypothetical shifted-gamma tape variant
    //
    // What would the max-diff be if tape.rs rms_norm_forward used shifted
    // gamma xi * (1+wi) * inv_rms instead of xi * wi * inv_rms?
    // Computed INLINE IN THIS TEST — production tape.rs is NOT modified.
    // ===================================================================

    #[test]
    fn test3_hypothetical_shifted_gamma_tape() {
        let hidden = 64usize;
        let eps = 1e-6f32;
        let mut rng = 0xF00D_CAFE_u64;

        let x = rand_vec(&mut rng, hidden, 1.0);
        let w = rand_vec(&mut rng, hidden, 0.5);

        // Real (baseline truth): shifted gamma xi*(1+wi)*inv_rms
        let mut real_out = x.clone();
        real_rms_norm_shifted(&mut real_out, &w, hidden, eps);

        // Current tape.rs: plain gamma xi*wi*inv_rms
        let (current_tape_out, _) = rms_norm_forward(&x, &w, eps);

        // Hypothetical shifted tape: xi*(1+wi)*inv_rms — computed inline
        let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / hidden as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let shifted_tape_out: Vec<f32> = x
            .iter()
            .zip(w.iter())
            .map(|(xi, wi)| xi * (1.0 + wi) * inv_rms)
            .collect();

        let diff_current = max_abs_diff(&current_tape_out, &real_out);
        let diff_shifted = max_abs_diff(&shifted_tape_out, &real_out);

        println!("=== TEST 3: Hypothetical shifted-gamma tape ===");
        println!("  hidden={hidden}, eps={eps}");
        println!("  MEASURED max-diff(current tape vs real):   {diff_current:.6e}");
        println!("  MEASURED max-diff(shifted tape vs real):   {diff_shifted:.6e}");
        if diff_shifted > 0.0 {
            println!("  Reduction from fix: {:.1}x", diff_current / diff_shifted);
        } else {
            println!("  Reduction from fix: inf (shifted tape IS the real formula)");
        }
        println!("  NOTE: production tape.rs was NOT modified; this is a what-if measurement.");
        println!(
            "  Verification: shifted_tape[0..4]: {:?}",
            &shifted_tape_out[..4]
        );
        println!("  Verification: real_out[0..4]:      {:?}", &real_out[..4]);
        println!();
    }

    // ===================================================================
    // TEST 4: Real-model LoRA forward parity
    //
    // Proves that the materialised GQA forward (gqa_forward_with_cache)
    // matches the REAL loaded Qwen3.5-0.8B model at layer 23, WITH an
    // identical nonzero LoRA on q_proj and v_proj injected into both paths.
    //
    // This closes the "self-consistent gradcheck" blind spot: prior
    // gradchecks compared the materialised forward to itself; this
    // compares it to the actual loaded model running real weights.
    //
    // Design:
    //   1. Load the model from disk (skip gracefully if absent).
    //   2. Build a nonzero LoRA for layer 23 (q_proj + v_proj only).
    //   3. Inject LoRA into the real forward via a test-local LoraHook.
    //   4. Run one token at position 0 through the real model, capturing
    //      h_in (pre-input-layernorm residual) and attn_out (gated o_proj).
    //   5. Apply input RMSNorm to h_in using the real layer-23 weights,
    //      then call gqa_forward_with_cache with the same LoRA and identity
    //      RoPE tables (position 0: cos=1, sin=0).
    //   6. Assert max-abs-diff < 1e-3.
    //
    // Skip condition: model dir absent → print SKIP and return without fail.
    // Feature gate: #[cfg(all(feature = "train-backward", feature = "f16"))]
    //   (f16 enables BF16 dequant in from_safetensors).
    // ===================================================================

    #[cfg(feature = "f16")]
    #[test]
    fn test4_real_model_lora_forward_parity_layer23() {
        use lattice_inference::lora_hook::LoraHook;
        use lattice_inference::model::qwen35::Qwen35Model;

        // ------------------------------------------------------------------
        // 1. Resolve model directory (env-var override with HF cache fallback)
        // ------------------------------------------------------------------
        let model_dir = std::env::var("LATTICE_QWEN35_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
                std::path::PathBuf::from(home)
                    .join(".cache/huggingface/hub")
                    .join("models--Qwen--Qwen3.5-0.8B")
                    .join("snapshots")
                    .join("2fc06364715b967f1860aea9cf38778875588b17")
            });

        if !model_dir.is_dir() {
            println!(
                "SKIP: model not found at {} — set LATTICE_QWEN35_MODEL_DIR to run",
                model_dir.display()
            );
            return;
        }

        // ------------------------------------------------------------------
        // 2. Load model and read real dims/eps from config
        // ------------------------------------------------------------------
        let mut model =
            Qwen35Model::from_safetensors(&model_dir).expect("failed to load Qwen3.5-0.8B");

        let cfg = model.config();
        let hidden = cfg.hidden_size;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let rope_dim = cfg.rope_dim();
        let eps = cfg.rms_norm_eps;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        println!(
            "=== TEST 4: real-model LoRA forward parity (materialised vs loaded Qwen3.5 layer-23) ==="
        );
        println!(
            "  dims: hidden={hidden} q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim}"
        );
        println!("        q_dim={q_dim} kv_dim={kv_dim} rope_dim={rope_dim}");
        println!("  rms_norm_eps={eps:.2e}  (from model config — not hardcoded)");

        // ------------------------------------------------------------------
        // 3. Build nonzero LoRA for layer 23 only: q_proj + v_proj
        //
        //    q_proj LoRA shape: A=[rank, hidden], B=[2*q_dim, rank]  (q_proj
        //      output is 2*q_dim due to per-head interleaved [Q|gate] layout)
        //    v_proj LoRA shape: A=[rank, hidden], B=[kv_dim, rank]
        // ------------------------------------------------------------------
        let lora_rank = 4usize;
        let lora_scale = 0.03f32;
        let mut rng = 0x1234_ABCD_5678_EF01_u64;

        let lora_a_q = rand_vec(&mut rng, lora_rank * hidden, 0.03);
        let lora_b_q = rand_vec(&mut rng, 2 * q_dim * lora_rank, 0.03);
        let lora_a_v = rand_vec(&mut rng, lora_rank * hidden, 0.03);
        let lora_b_v = rand_vec(&mut rng, kv_dim * lora_rank, 0.03);

        // ------------------------------------------------------------------
        // 4. Test-local LoraHook: injects LoRA for (layer=23, q_proj)
        //    and (layer=23, v_proj); no-op for everything else.
        //
        //    Convention matches gqa_forward_with_cache and production
        //    forward.rs: output += scale * B @ (A @ x), row-major, A then B.
        // ------------------------------------------------------------------
        struct TestLoraHook {
            layer: usize,
            rank: usize,
            scale: f32,
            a_q: Vec<f32>,
            b_q: Vec<f32>,
            a_v: Vec<f32>,
            b_v: Vec<f32>,
        }

        impl LoraHook for TestLoraHook {
            fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
                if layer_idx != self.layer {
                    return;
                }
                let (a, b, out_dim) = match module {
                    "q_proj" => (&self.a_q, &self.b_q, output.len()),
                    "v_proj" => (&self.a_v, &self.b_v, output.len()),
                    _ => return,
                };
                let rank = self.rank;
                let in_dim = x.len();
                // h = A @ x  [rank]
                let mut h = vec![0.0f32; rank];
                for r in 0..rank {
                    h[r] = a[r * in_dim..(r + 1) * in_dim]
                        .iter()
                        .zip(x.iter())
                        .map(|(ai, xi)| ai * xi)
                        .sum();
                }
                // output += scale * B @ h  [out_dim]
                for i in 0..out_dim {
                    let acc: f32 = b[i * rank..(i + 1) * rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(bi, hi)| bi * hi)
                        .sum();
                    output[i] += self.scale * acc;
                }
            }
        }

        model.set_lora(Box::new(TestLoraHook {
            layer: 23,
            rank: lora_rank,
            scale: lora_scale,
            a_q: lora_a_q.clone(),
            b_q: lora_b_q.clone(),
            a_v: lora_a_v.clone(),
            b_v: lora_b_v.clone(),
        }));

        // ------------------------------------------------------------------
        // 5. Run real forward for a single token at position 0.
        //    capture_attn_io runs forward_step for each token in sequence;
        //    with a single token [100], position=0, RoPE is identity (cos=1,
        //    sin=0 for all dims since theta*0=0).
        // ------------------------------------------------------------------
        let tokens: Vec<u32> = vec![100];
        let (h_in, captured_attn_out) = model
            .capture_attn_io(&tokens, 23)
            .expect("capture_attn_io failed for layer 23");

        assert_eq!(
            h_in.len(),
            hidden,
            "h_in length mismatch: got {} expected {hidden}",
            h_in.len()
        );
        assert_eq!(
            captured_attn_out.len(),
            hidden,
            "captured_attn_out length mismatch: got {} expected {hidden}",
            captured_attn_out.len()
        );

        // ------------------------------------------------------------------
        // 6. Get real layer-23 weights
        // ------------------------------------------------------------------
        let (w_q, w_k, w_v, w_o, q_norm_w, k_norm_w, pre_attn_norm, _post, _g, _u, _d) = model
            .gqa_layer_weights(23)
            .expect("layer 23 is a Full+Dense GQA layer");

        // ------------------------------------------------------------------
        // 7. Recompute input-layernorm: normed = rms_norm(h_in, pre_attn_norm)
        //    using real shifted gamma (1+gamma)*inv_rms, matching qwen35_rms_norm.
        //    The materialised forward expects already-normed input.
        // ------------------------------------------------------------------
        let mut normed_input = h_in.clone();
        real_rms_norm_shifted(&mut normed_input, pre_attn_norm, hidden, eps);

        // ------------------------------------------------------------------
        // 8. Build identity RoPE tables for position 0: cos=1, sin=0.
        //    rope_dim/2 entries per position; seq_len=1.
        // ------------------------------------------------------------------
        let half_rope = rope_dim / 2;
        let cos_table = vec![1.0f32; half_rope]; // cos(0) = 1
        let sin_table = vec![0.0f32; half_rope]; // sin(0) = 0

        // ------------------------------------------------------------------
        // 9. Run materialised forward (gqa_forward_with_cache)
        //    with the same LoRA arrays and real weights.
        // ------------------------------------------------------------------
        let (materialised_out, _cache) = gqa_forward_with_cache(
            &normed_input,
            w_q,
            w_k,
            w_v,
            w_o,
            q_norm_w,
            k_norm_w,
            Some(&lora_a_q),
            Some(&lora_b_q),
            Some(&lora_a_v),
            Some(&lora_b_v),
            lora_rank,
            lora_scale,
            1, // seq_len
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            &cos_table,
            &sin_table,
            eps,
        );

        // ------------------------------------------------------------------
        // 10. Localisation baseline: run materialised forward with zero LoRA
        //     to isolate the LoRA-injection contribution.
        // ------------------------------------------------------------------
        let zero_a_q = vec![0.0f32; lora_rank * hidden];
        let zero_b_q = vec![0.0f32; 2 * q_dim * lora_rank];
        let zero_a_v = vec![0.0f32; lora_rank * hidden];
        let zero_b_v = vec![0.0f32; kv_dim * lora_rank];

        // Materialised forward with zero LoRA (M0).
        let (materialised_no_lora, _) = gqa_forward_with_cache(
            &normed_input,
            w_q,
            w_k,
            w_v,
            w_o,
            q_norm_w,
            k_norm_w,
            Some(&zero_a_q),
            Some(&zero_b_q),
            Some(&zero_a_v),
            Some(&zero_b_v),
            lora_rank,
            lora_scale,
            1,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            &cos_table,
            &sin_table,
            eps,
        );

        // True no-LoRA real baseline (R0): re-capture the real model with a
        // zero-LoRA hook so d_no_lora compares materialised-no-LoRA (M0) vs
        // real-no-LoRA (R0), isolating base f32 divergence from the LoRA delta.
        // Comparing M0 against the with-LoRA capture (R1) would conflate the
        // base gap with the LoRA-delta magnitude (they coincide numerically).
        model.set_lora(Box::new(TestLoraHook {
            layer: 23,
            rank: lora_rank,
            scale: lora_scale,
            a_q: zero_a_q,
            b_q: zero_b_q,
            a_v: zero_a_v,
            b_v: zero_b_v,
        }));
        let (_h_in_no_lora, captured_attn_out_no_lora) = model
            .capture_attn_io(&tokens, 23)
            .expect("capture_attn_io (no-LoRA baseline) failed for layer 23");

        // ------------------------------------------------------------------
        // 11. Measure and report
        // ------------------------------------------------------------------
        let d = max_abs_diff(&materialised_out, &captured_attn_out);
        let d_no_lora = max_abs_diff(&materialised_no_lora, &captured_attn_out_no_lora);
        let lora_contribution = max_abs_diff(&materialised_out, &materialised_no_lora);

        println!();
        println!(
            "  Sample materialised_out[0..4]: {:?}",
            &materialised_out[..4.min(hidden)]
        );
        println!(
            "  Sample captured_attn_out[0..4]: {:?}",
            &captured_attn_out[..4.min(hidden)]
        );
        println!();
        println!("  Localisation:");
        println!("    max-diff(materialised vs real), NO LoRA:   {d_no_lora:.6e}");
        println!("    LoRA delta contribution (mat_lora - mat_no_lora): {lora_contribution:.6e}");
        println!("    MEASURED max-diff(materialised vs real model): {d:.6e}");

        if d < 1e-3 {
            println!(
                "  [PASS] max-diff < 1e-3: materialised GQA forward matches real model with LoRA."
            );
        } else {
            println!("  [FAIL] max-diff >= 1e-3: divergence detected.");
            println!("  Diagnosis hints:");
            if d_no_lora < 1e-3 && d >= 1e-3 {
                println!("    LoRA injection mismatch (no-LoRA passes, LoRA fails).");
            } else if d_no_lora >= 1e-3 {
                println!(
                    "    Base forward divergence (no-LoRA fails); check norm eps, o_proj LoRA."
                );
            }
        }
        println!();

        assert!(
            d < 1e-3,
            "TEST 4 FAILED: max-abs-diff(materialised vs real-model) = {d:.6e} >= 1e-3. \
             No-LoRA baseline diff = {d_no_lora:.6e}. LoRA contribution = {lora_contribution:.6e}."
        );
    }
}
