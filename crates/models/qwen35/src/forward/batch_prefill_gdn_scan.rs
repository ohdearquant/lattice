//! The GatedDeltaNet recurrence of the batched prompt prefill.
//!
//! Each value head owns its recurrent state and reads only its own slices of the
//! per-token rows, so the prefill runs the causal conv over the whole prompt
//! first and then hands every head all of the prompt's tokens in one parallel
//! pass: one fork-join per layer, with each head's state staying on one thread.
//! Every value is computed by the same operations, in the same order, as the
//! token-by-token loop, which stays for the state observer, since it watches
//! every head after every token.

use super::compute_decay_gate_prefill;
use crate::attention::gdn::{
    GatedDeltaNetState, GatedDeltaNetWeights, gated_rms_norm, l2_normalize_vec,
};
use lattice_arch::qwen35_config::Qwen35Config;
use rayon::prelude::*;

#[cfg(any(test, feature = "gdn-state-observer"))]
use super::apply_causal_conv1d_prefill;

/// The conv pass splits the prompt into at most this many blocks of rows.
pub(super) const CONV_BLOCKS: usize = 32;
/// The fewest prompt rows the conv pass gives one block.
const CONV_MIN_BLOCK_ROWS: usize = 64;

/// One layer's per-token rows, each `[seq_len, width]`.
pub(super) struct ScanRows<'a> {
    /// The qkv projection. The by-head scan replaces it with the conv output.
    pub(super) qkv: &'a mut [f32],
    /// The output gate. The by-head scan replaces it with the scan output.
    pub(super) z: &'a mut [f32],
    pub(super) beta: &'a [f32],
    pub(super) alpha: &'a [f32],
    /// The token-by-token loop writes the scan output here. The by-head scan
    /// uses it as head-major workspace.
    pub(super) out: &'a mut [f32],
}

/// Workspaces: one slice per value head for each vector, and the conv's.
pub(super) struct ScanWork<'a> {
    /// One token's conv output, for the token-by-token loop: `[qkv_dim]`.
    #[cfg_attr(not(any(test, feature = "gdn-state-observer")), allow(dead_code))]
    pub(super) conv_token: &'a mut [f32],
    /// The input rows before each conv block, then the next conv state:
    /// `[(CONV_BLOCKS + 1) * (kernel - 1) * qkv_dim]`.
    pub(super) conv_history: &'a mut [f32],
    /// One accumulator row per conv block: `[CONV_BLOCKS * qkv_dim]`.
    pub(super) conv_acc: &'a mut [f32],
    pub(super) q: &'a mut [f32],
    pub(super) k: &'a mut [f32],
    pub(super) kv_mem: &'a mut [f32],
    pub(super) delta: &'a mut [f32],
    pub(super) out_head: &'a mut [f32],
}

#[derive(Clone, Copy)]
struct Dims {
    value_heads: usize,
    ratio: usize,
    key_dim: usize,
    value_dim: usize,
    qkv_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    q_total: usize,
    v_offset: usize,
    s_size: usize,
    retrieval_scale: f32,
    eps: f32,
}

impl Dims {
    fn new(cfg: &Qwen35Config) -> Self {
        let num_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let q_total = num_heads * key_dim;
        let k_total = num_heads * key_dim;
        Self {
            value_heads,
            ratio: value_heads / num_heads,
            key_dim,
            value_dim,
            qkv_dim: cfg.linear_qkv_dim(),
            output_dim: cfg.linear_output_dim(),
            kernel_size: cfg.linear_conv_kernel_dim,
            q_total,
            v_offset: q_total + k_total,
            s_size: key_dim * value_dim,
            retrieval_scale: 1.0 / (key_dim as f32).sqrt(),
            eps: cfg.rms_norm_eps,
        }
    }
}

/// Run the recurrence over the prompt one head at a time, all heads in
/// parallel, leaving the scan output token-major in `rows.z`.
///
/// `rows.qkv` and `rows.out` are overwritten. `state` advances exactly as the
/// token-by-token loop advances it.
#[cfg_attr(feature = "gdn-state-observer", allow(dead_code))]
pub(super) fn scan_by_head(
    cfg: &Qwen35Config,
    weights: &GatedDeltaNetWeights,
    state: &mut GatedDeltaNetState,
    seq_len: usize,
    rows: ScanRows<'_>,
    work: ScanWork<'_>,
) {
    let d = Dims::new(cfg);
    let ScanRows {
        qkv,
        z,
        beta,
        alpha,
        out,
    } = rows;
    let qkv = &mut qkv[..seq_len * d.qkv_dim];
    conv_in_place(
        &d,
        &weights.conv1d_weight,
        &mut state.conv_buffer,
        qkv,
        work.conv_history,
        work.conv_acc,
        seq_len,
    );
    let qkv = &*qkv;
    let gate = &*z;
    let head_rows = seq_len * d.value_dim;
    let norm_weight = &weights.norm_weight[..d.value_dim];
    // `zip` stops at the shortest operand, so each one must split into
    // exactly `value_heads` chunks or the trailing heads are skipped; the
    // transpose below reads the same per-head layout.
    debug_assert_eq!(d.output_dim, d.value_heads * d.value_dim);

    state.s_matrices[..d.value_heads * d.s_size]
        .par_chunks_mut(d.s_size)
        .zip(out[..d.value_heads * head_rows].par_chunks_mut(head_rows))
        .zip(work.q[..d.value_heads * d.key_dim].par_chunks_mut(d.key_dim))
        .zip(work.k[..d.value_heads * d.key_dim].par_chunks_mut(d.key_dim))
        .zip(work.kv_mem[..d.output_dim].par_chunks_mut(d.value_dim))
        .zip(work.delta[..d.output_dim].par_chunks_mut(d.value_dim))
        .zip(work.out_head[..d.output_dim].par_chunks_mut(d.value_dim))
        .enumerate()
        .for_each(
            |(h, ((((((s, head_out), q_tmp), k_tmp), kv_mem), delta), out_head))| {
                let k_head = h / d.ratio;
                let q_start = k_head * d.key_dim;
                let k_start = d.q_total + k_head * d.key_dim;
                let v_start = d.v_offset + h * d.value_dim;
                let z_start = h * d.value_dim;
                let value_dim = d.value_dim;

                for t in 0..seq_len {
                    let conv = &qkv[t * d.qkv_dim..(t + 1) * d.qkv_dim];
                    q_tmp.copy_from_slice(&conv[q_start..q_start + d.key_dim]);
                    k_tmp.copy_from_slice(&conv[k_start..k_start + d.key_dim]);
                    let v_vec = &conv[v_start..v_start + value_dim];

                    l2_normalize_vec(q_tmp);
                    l2_normalize_vec(k_tmp);

                    let g = compute_decay_gate_prefill(
                        weights.a_log[h],
                        alpha[t * d.value_heads + h],
                        weights.dt_bias[h],
                    );

                    // Decay each state row, then add it into kv_mem = S^T @ k.
                    kv_mem.fill(0.0);
                    for (i, &k_i) in k_tmp.iter().enumerate() {
                        let row = &mut s[i * value_dim..(i + 1) * value_dim];
                        for (mem, r) in kv_mem.iter_mut().zip(row.iter_mut()) {
                            *r *= g;
                            *mem += *r * k_i;
                        }
                    }

                    // delta = (v - kv_mem) * beta
                    let beta_h = beta[t * d.value_heads + h];
                    for (dl, (&v_j, &mem_j)) in
                        delta.iter_mut().zip(v_vec.iter().zip(kv_mem.iter()))
                    {
                        *dl = (v_j - mem_j) * beta_h;
                    }

                    // Add outer(k, delta) to each state row, then add the row into
                    // output = S^T @ q.
                    out_head.fill(0.0);
                    for (i, (&k_i, &q_i)) in k_tmp.iter().zip(q_tmp.iter()).enumerate() {
                        let row = &mut s[i * value_dim..(i + 1) * value_dim];
                        for ((r, &dl), o) in
                            row.iter_mut().zip(delta.iter()).zip(out_head.iter_mut())
                        {
                            *r += k_i * dl;
                            *o += *r * q_i;
                        }
                    }
                    for o in out_head.iter_mut() {
                        *o *= d.retrieval_scale;
                    }

                    let z_row = &gate[t * d.output_dim + z_start..][..value_dim];
                    gated_rms_norm(
                        out_head,
                        z_row,
                        norm_weight,
                        &mut head_out[t * value_dim..(t + 1) * value_dim],
                        d.eps,
                    );
                }
            },
        );

    // Back to token-major rows, into the gate rows the heads no longer read.
    let out = &*out;
    z[..seq_len * d.output_dim]
        .par_chunks_mut(d.output_dim)
        .enumerate()
        .for_each(|(t, row)| {
            for (h, dst) in row.chunks_exact_mut(d.value_dim).enumerate() {
                let src = h * head_rows + t * d.value_dim;
                dst.copy_from_slice(&out[src..src + d.value_dim]);
            }
        });
}

/// Replace each qkv row with SiLU of its causal depthwise conv, and advance
/// `conv_state` past the prompt.
///
/// The prompt is cut into blocks of rows that run in parallel. A block works
/// from its last row back to its first, so the input rows it still needs are
/// never overwritten, and the input rows before it are copied out first.
fn conv_in_place(
    d: &Dims,
    conv_weight: &[f32],
    conv_state: &mut [f32],
    qkv: &mut [f32],
    conv_history: &mut [f32],
    conv_acc: &mut [f32],
    seq_len: usize,
) {
    let qkv_dim = d.qkv_dim;
    let kernel = d.kernel_size;
    let buf_len = kernel.saturating_sub(1);
    let block_rows = seq_len.div_ceil(CONV_BLOCKS).max(CONV_MIN_BLOCK_ROWS);
    let blocks = seq_len.div_ceil(block_rows);
    let history_len = buf_len * qkv_dim;
    let (history, rest) = conv_history.split_at_mut(CONV_BLOCKS * history_len);
    let next_state = &mut rest[..history_len];

    {
        // Input row `i` of the prompt; rows before it come from the conv state,
        // which holds the last `buf_len` inputs, oldest first, per channel.
        let input = |i: isize, ch: usize| -> f32 {
            if i >= 0 {
                qkv[i as usize * qkv_dim + ch]
            } else {
                conv_state[ch * buf_len + (i + buf_len as isize) as usize]
            }
        };
        for b in 0..blocks {
            let first = (b * block_rows) as isize - buf_len as isize;
            for r in 0..buf_len {
                let dst = &mut history[(b * buf_len + r) * qkv_dim..][..qkv_dim];
                for (ch, v) in dst.iter_mut().enumerate() {
                    *v = input(first + r as isize, ch);
                }
            }
        }
        let first = seq_len as isize - buf_len as isize;
        for ch in 0..qkv_dim {
            for j in 0..buf_len {
                next_state[ch * buf_len + j] = input(first + j as isize, ch);
            }
        }
    }

    let history = &*history;
    qkv.par_chunks_mut(block_rows * qkv_dim)
        .zip(conv_acc[..blocks * qkv_dim].par_chunks_mut(qkv_dim))
        .enumerate()
        .for_each(|(b, (block, acc))| {
            let before = &history[b * history_len..(b + 1) * history_len];
            for t in (0..block.len() / qkv_dim).rev() {
                let (earlier, row) = block.split_at_mut(t * qkv_dim);
                let row = &mut row[..qkv_dim];
                acc.fill(0.0);
                for j in 0..buf_len {
                    let prev = if t + j >= buf_len {
                        &earlier[(t + j - buf_len) * qkv_dim..][..qkv_dim]
                    } else {
                        &before[(t + j) * qkv_dim..][..qkv_dim]
                    };
                    for (ch, (a, &x)) in acc.iter_mut().zip(prev).enumerate() {
                        *a += x * conv_weight[ch * kernel + j];
                    }
                }
                for (ch, (&a, x)) in acc.iter().zip(row.iter_mut()).enumerate() {
                    let sum = a + *x * conv_weight[ch * kernel + buf_len];
                    *x = sum / (1.0 + (-sum).exp());
                }
            }
        });
    conv_state[..history_len].copy_from_slice(next_state);
}

/// Run the recurrence one prompt position at a time, every head in parallel
/// at each position, writing the scan output token-major into `rows.out`.
#[cfg(any(test, feature = "gdn-state-observer"))]
pub(super) fn scan_per_token(
    cfg: &Qwen35Config,
    weights: &GatedDeltaNetWeights,
    state: &mut GatedDeltaNetState,
    seq_len: usize,
    rows: ScanRows<'_>,
    work: ScanWork<'_>,
    layer_idx: usize,
) {
    #[cfg(not(feature = "gdn-state-observer"))]
    let _ = layer_idx;
    let Dims {
        value_heads,
        ratio,
        key_dim,
        value_dim,
        qkv_dim,
        output_dim,
        kernel_size,
        q_total,
        v_offset,
        s_size,
        retrieval_scale,
        ..
    } = Dims::new(cfg);

    // Sequential recurrence over prompt positions.
    for t in 0..seq_len {
        let qkv_row = &rows.qkv[t * qkv_dim..(t + 1) * qkv_dim];
        let z_row = &rows.z[t * output_dim..(t + 1) * output_dim];
        let beta_row = &rows.beta[t * value_heads..(t + 1) * value_heads];
        let alpha_row = &rows.alpha[t * value_heads..(t + 1) * value_heads];
        let out_row = &mut rows.out[t * output_dim..(t + 1) * output_dim];

        apply_causal_conv1d_prefill(
            qkv_row,
            &mut state.conv_buffer,
            &weights.conv1d_weight,
            &mut work.conv_token[..qkv_dim],
            qkv_dim,
            kernel_size,
        );

        // SiLU(conv_output).
        for v in &mut work.conv_token[..qkv_dim] {
            *v = *v / (1.0 + (-*v).exp());
        }

        // Process all heads. Each head owns its recurrent state, its output
        // slice and its temporaries, so the heads run on the rayon pool and
        // every value is computed by the same operations, in the same order,
        // as one head at a time.
        let conv = &work.conv_token[..qkv_dim];
        let norm_weight = &weights.norm_weight[..value_dim];
        let eps = cfg.rms_norm_eps;
        // `zip` stops at the shortest operand, so each one must split into
        // exactly `value_heads` chunks or the trailing heads are skipped.
        debug_assert_eq!(output_dim, value_heads * value_dim);
        debug_assert_eq!(out_row.len(), value_heads * value_dim);
        state.s_matrices[..value_heads * s_size]
            .par_chunks_mut(s_size)
            .zip(work.out_head[..output_dim].par_chunks_mut(value_dim))
            .zip(work.q[..value_heads * key_dim].par_chunks_mut(key_dim))
            .zip(work.k[..value_heads * key_dim].par_chunks_mut(key_dim))
            .zip(work.kv_mem[..output_dim].par_chunks_mut(value_dim))
            .zip(work.delta[..output_dim].par_chunks_mut(value_dim))
            .zip(out_row.par_chunks_mut(value_dim))
            .enumerate()
            .for_each(
                |(h, ((((((s, out_head), q_tmp), k_tmp), kv_mem), delta), out))| {
                    let k_head = h / ratio;
                    let q_start = k_head * key_dim;
                    let k_start = q_total + k_head * key_dim;
                    let v_start = v_offset + h * value_dim;

                    q_tmp.copy_from_slice(&conv[q_start..q_start + key_dim]);
                    k_tmp.copy_from_slice(&conv[k_start..k_start + key_dim]);
                    let v_vec = &conv[v_start..v_start + value_dim];

                    l2_normalize_vec(q_tmp);
                    l2_normalize_vec(k_tmp);

                    let g = compute_decay_gate_prefill(
                        weights.a_log[h],
                        alpha_row[h],
                        weights.dt_bias[h],
                    );

                    // Decay the recurrent state.
                    for val in s.iter_mut() {
                        *val *= g;
                    }

                    // kv_mem = S^T @ k
                    kv_mem.fill(0.0);
                    for (i, &k_i) in k_tmp.iter().enumerate() {
                        let row = &s[i * value_dim..(i + 1) * value_dim];
                        for (mem, &r) in kv_mem.iter_mut().zip(row) {
                            *mem += r * k_i;
                        }
                    }

                    // delta = (v - kv_mem) * beta
                    let beta_h = beta_row[h];
                    for (d, (&v_j, &mem_j)) in delta.iter_mut().zip(v_vec.iter().zip(kv_mem.iter()))
                    {
                        *d = (v_j - mem_j) * beta_h;
                    }

                    // S += outer(k, delta)
                    for (i, &k_i) in k_tmp.iter().enumerate() {
                        let row = &mut s[i * value_dim..(i + 1) * value_dim];
                        for (r, &d) in row.iter_mut().zip(delta.iter()) {
                            *r += k_i * d;
                        }
                    }

                    // output = S^T @ q / sqrt(key_dim), accumulated row by row so
                    // the inner loop reads S contiguously; each output still sums
                    // its terms in ascending i, so the result is unchanged bit for
                    // bit.
                    out_head.fill(0.0);
                    for (i, &q_i) in q_tmp.iter().enumerate() {
                        let row = &s[i * value_dim..(i + 1) * value_dim];
                        for (o, &r) in out_head.iter_mut().zip(row) {
                            *o += r * q_i;
                        }
                    }
                    for o in out_head.iter_mut() {
                        *o *= retrieval_scale;
                    }

                    // Gated RMSNorm for this head.
                    // Note: pass norm_weight[..value_dim] to match the decode path
                    // in gated_delta_net_step, which passes the full norm_weight
                    // and the function only reads the first `dim` elements.
                    let z_start = h * value_dim;
                    gated_rms_norm(
                        out_head,
                        &z_row[z_start..z_start + value_dim],
                        norm_weight,
                        out,
                        eps,
                    );
                },
            );

        #[cfg(feature = "gdn-state-observer")]
        crate::gdn_state_observer::observe(layer_idx, t, state, cfg);
    }
}

#[cfg(test)]
#[path = "batch_prefill_gdn_scan_tests.rs"]
mod tests;
