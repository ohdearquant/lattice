//! Self-contained integration test for Native Sparse Attention (NSA).
//!
//! # Reference
//!
//! ADR-042 (`docs/adr/ADR-042-native-sparse-attention.md`) is the authoritative spec.
//! The naive oracle below (`ref_native_sparse_attention`) is a direct transcription of
//! the paper's equations (Yuan et al., arXiv:2502.11089, Eq. 5, 7–12). It shares NO code
//! with the kernel body — but, like the kernel, it transcribes its spec from the paper
//! via ADR-042, so a kernel-vs-oracle parity test is implementation-independent, *not*
//! specification-independent. (Round-1 review caught an Eq. 9 sign error that both had
//! copied from this ADR.) The Eq. 9 index is therefore also checked against the paper
//! directly by `aggregate_selection_importance`'s hand-computed unit test in the kernel
//! module — that test, not this parity oracle, is the spec-level anchor for Eq. 9.
//!
//! # Algorithm summary (oracle follows this exactly)
//!
//! Per paper §3.3.3 the three branches use **independent K/V** — the kernel and this
//! oracle take 8 activation buffers: `q` (non-RoPE) and `q_rope`; `k_cmp` (non-RoPE),
//! `k_slc`, `k_win` (RoPE'd); `v_cmp`, `v_slc`, `v_win` (RoPE-free).
//!
//! For each query position `t`:
//!
//! **Compression branch** (uses non-RoPE `q` / `k_cmp` / `v_cmp`, ADR-042 §4)
//!   - Compression block `i` covers raw-K tokens `[i*d, i*d+l)`.
//!   - Intra-block PE (learned, per KV-head) is added to each block before φ.
//!   - φ: Linear(phi_in→phi_in) → ReLU → Linear(phi_in→head_dim). Shared weights,
//!     per-KV-head output.
//!   - Causal validity: block `i` is valid for `t` iff `i*d + l <= t+1` (all `l`
//!     tokens of the block have been seen).
//!   - Compression attention: softmax(q_t · ck_valid^T / √head_dim) → p_t^cmp.
//!     Masked positions contribute 0.
//!   - o_t^cmp = p_t^cmp · cv_valid.
//!
//! **Selection branch** (uses RoPE'd `q_rope` / `k_slc` / `v_slc`, ADR-042 §5–7)
//!   - Selection block `j` covers tokens `[j*l', (j+1)*l')`.
//!   - Valid for `t` iff `j*l' < t+1` (at least one token seen).
//!   - Eq. 9 importance (paper-verbatim index, **minus** both `m` and `n`):
//!     `p_slc[j] = Σ_{m=0}^{cps-1} Σ_{n=0}^{ipc-1} p_cmp[cps*j - m - n]`
//!     where `cps = l'/d`, `ipc = l/d`. A compression index outside `[0, n_cblocks)` —
//!     negative for small `j`, or causally invalid — contributes 0.
//!   - Eq. 10 GQA group sum: sum importance across all query heads in a KV-head group.
//!   - Top-n forced: always include block 0 (initial) and the 2 highest-indexed valid
//!     blocks (local); remaining n-3 are top-scored from the rest.
//!   - Gather ≤ n*l' tokens, apply token-level causal mask, standard softmax attn.
//!
//! **Sliding-window branch** (uses RoPE'd `q_rope` / `k_win` / `v_win`, ADR-042 §8)
//!   - Tokens `[max(0, t-w+1), t]` inclusive.
//!
//! **Gating** (ADR-042 §9)
//!   - g = sigmoid(g_proj_w · x_t + g_proj_b), rows ordered [h=0,cmp],[h=0,slc],[h=0,win],...
//!   - o_t = g_cmp * o_cmp + g_slc * o_slc + g_win * o_win  (un-normalized).
//!
//! Run:
//!   cargo test -p lattice-inference --test native_sparse_attention_test

use lattice_inference::attention::native_sparse::{
    NsaConfig, NsaScratch, NsaWeights, apply_native_sparse_attention,
};

// ===================================================================
// Deterministic PRNG — xorshift64, same generator as the other tests
// ===================================================================

fn det_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 7;
        state ^= state >> 9;
        state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mantissa = ((state >> 41) as u32) & 0x007f_ffff;
        let x = f32::from_bits(0x3f80_0000 | mantissa) - 1.5;
        out.push(x);
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ===================================================================
// Helpers: sigmoid, softmax, φ MLP
// ===================================================================

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax in-place over a slice. Empty slice is a no-op.
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Apply φ MLP to one flattened block vector.
///
/// φ: Linear(phi_in → phi_in) → ReLU → Linear(phi_in → head_dim)
/// w1: [phi_in, phi_in] row-major, b1: [phi_in]
/// w2: [head_dim, phi_in] row-major, b2: [head_dim]
fn apply_phi(
    input: &[f32], // length phi_in = l * head_dim
    w1: &[f32],    // [phi_in, phi_in]
    b1: &[f32],    // [phi_in]
    w2: &[f32],    // [head_dim, phi_in]
    b2: &[f32],    // [head_dim]
    phi_in: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Layer 1: hidden = relu(w1 * input + b1), shape [phi_in]
    let mut hidden = vec![0.0_f32; phi_in];
    for r in 0..phi_in {
        let mut acc = b1[r];
        for c in 0..phi_in {
            acc += w1[r * phi_in + c] * input[c];
        }
        hidden[r] = acc.max(0.0); // ReLU
    }

    // Layer 2: out = w2 * hidden + b2, shape [head_dim]
    let mut out = vec![0.0_f32; head_dim];
    for r in 0..head_dim {
        let mut acc = b2[r];
        for c in 0..phi_in {
            acc += w2[r * phi_in + c] * hidden[c];
        }
        out[r] = acc;
    }
    out
}

// ===================================================================
// Dense causal attention reference (used by test_window_only_equals_dense)
//
// Standard scaled-dot-product causal attention.
// q_rope, k_rope: [seq_len, num_heads * head_dim]
// v:              [seq_len, num_kv_heads * head_dim]
// output:         [seq_len, num_heads * head_dim]
// ===================================================================

fn ref_dense_causal_attention(
    q_rope: &[f32],
    k_rope: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_rep = num_heads / num_kv_heads;
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0_f32; seq_len * num_heads * head_dim];

    for h in 0..num_heads {
        let kv_h = h / n_rep;
        for t in 0..seq_len {
            // Dot with all causally-valid key positions 0..=t
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q_rope[q_off..q_off + head_dim];

            let mut scores = vec![0.0_f32; t + 1];
            for s in 0..=t {
                let k_off = s * num_kv_heads * head_dim + kv_h * head_dim;
                let k_row = &k_rope[k_off..k_off + head_dim];
                let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                scores[s] = dot * scale;
            }
            softmax_inplace(&mut scores);

            // Weighted sum of V
            let out_off = t * num_heads * head_dim + h * head_dim;
            for s in 0..=t {
                let v_off = s * num_kv_heads * head_dim + kv_h * head_dim;
                let v_row = &v[v_off..v_off + head_dim];
                let w = scores[s];
                for d in 0..head_dim {
                    out[out_off + d] += w * v_row[d];
                }
            }
        }
    }
    out
}

// ===================================================================
// Reference oracle: naive NSA from ADR-042 / paper equations
//
// This is a dead-simple, completely unoptimized transcription.
// Shares NO code with native_sparse.rs internals.
// ===================================================================

/// Naive reference implementation of Native Sparse Attention.
///
/// Inputs match `apply_native_sparse_attention` exactly.
/// Output: `[seq_len, num_heads * head_dim]`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn ref_native_sparse_attention(
    q_buf: &[f32],      // [seq_len, num_heads*head_dim]      non-RoPE Q (compression)
    q_rope_buf: &[f32], // [seq_len, num_heads*head_dim]      RoPE'd Q (selection, window)
    k_cmp_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   non-RoPE K, compression
    k_slc_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   RoPE'd K, selection
    k_win_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   RoPE'd K, window
    v_cmp_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   V, compression (RoPE-free)
    v_slc_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   V, selection (RoPE-free)
    v_win_buf: &[f32],  // [seq_len, num_kv_heads*head_dim]   V, window (RoPE-free)
    x_buf: &[f32],      // [seq_len, model_dim]
    weights: &NsaWeights,
    seq_len: usize,
    cfg: &NsaConfig,
) -> Vec<f32> {
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let l = cfg.compress_block;
    let d = cfg.compress_stride;
    let lp = cfg.select_block;
    let n_sel = cfg.num_selected;
    let win = cfg.window;
    let n_rep = cfg.n_rep();
    let phi_in = cfg.phi_in(); // l * head_dim

    let scale = (head_dim as f32).powf(-0.5);

    // ----------------------------------------------------------------
    // Model dim: inferred from g_proj_w shape [3*num_heads, model_dim]
    // ----------------------------------------------------------------
    let model_dim = weights.g_proj_w.len() / (3 * num_heads);

    // ----------------------------------------------------------------
    // Step 1: Build compressed K and V for each KV head.
    //
    // For KV head `kv_h` and block `i` (covering tokens [i*d, i*d+l)):
    //   1. Gather the l raw-K (or raw-V) token vectors.
    //   2. Add the intra-block PE (per KV-head, per intra-block-position).
    //   3. Flatten to phi_in, apply φ_k (or φ_v).
    //   4. Result: compressed vector of length head_dim.
    //
    // intra-block PE layout: [kv_h * l * head_dim + p * head_dim + d]
    // ----------------------------------------------------------------

    let n_cblocks = cfg.num_compress_blocks(seq_len);

    // ck[kv_h][i][head_dim], cv[kv_h][i][head_dim]
    let mut ck = vec![vec![vec![0.0_f32; head_dim]; n_cblocks]; num_kv_heads];
    let mut cv = vec![vec![vec![0.0_f32; head_dim]; n_cblocks]; num_kv_heads];

    for kv_h in 0..num_kv_heads {
        for i in 0..n_cblocks {
            let start_tok = i * d;
            // Flatten block with intra-block PE added
            let mut block_k = vec![0.0_f32; phi_in];
            let mut block_v = vec![0.0_f32; phi_in];
            for p in 0..l {
                let tok = start_tok + p;
                let pe_off = kv_h * l * head_dim + p * head_dim;
                let raw_k_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                let raw_v_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                for dim in 0..head_dim {
                    block_k[p * head_dim + dim] =
                        k_cmp_buf[raw_k_off + dim] + weights.k_intrablock_pos[pe_off + dim];
                    block_v[p * head_dim + dim] =
                        v_cmp_buf[raw_v_off + dim] + weights.v_intrablock_pos[pe_off + dim];
                }
            }

            // Apply φ_k
            ck[kv_h][i] = apply_phi(
                &block_k,
                &weights.phi_k_w1,
                &weights.phi_k_b1,
                &weights.phi_k_w2,
                &weights.phi_k_b2,
                phi_in,
                head_dim,
            );

            // Apply φ_v
            cv[kv_h][i] = apply_phi(
                &block_v,
                &weights.phi_v_w1,
                &weights.phi_v_b1,
                &weights.phi_v_w2,
                &weights.phi_v_b2,
                phi_in,
                head_dim,
            );
        }
    }

    // ----------------------------------------------------------------
    // Main loop: for each query position t, compute the three branches
    // and merge with gates.
    // ----------------------------------------------------------------

    // cps = l'/d, ipc = l/d (for Eq. 9)
    let cps = cfg.compress_per_select(); // l' / d
    let ipc = cfg.intra_per_compress(); // l / d

    let n_sblocks_total = cfg.num_select_blocks(seq_len);

    let mut output = vec![0.0_f32; seq_len * num_heads * head_dim];

    for t in 0..seq_len {
        // ---- Gate: g = sigmoid(g_proj_w · x_t + g_proj_b) ----
        // g_proj_w: [3*num_heads, model_dim] row-major
        // g_proj_b: [3*num_heads]
        // Row ordering: [h=0,cmp],[h=0,slc],[h=0,win],[h=1,cmp],[h=1,slc],[h=1,win],...
        let x_off = t * model_dim;
        let x_t = &x_buf[x_off..x_off + model_dim];

        let num_gate_rows = 3 * num_heads;
        let mut gates_raw = vec![0.0_f32; num_gate_rows];
        for row in 0..num_gate_rows {
            let mut acc = weights.g_proj_b[row];
            for dim in 0..model_dim {
                acc += weights.g_proj_w[row * model_dim + dim] * x_t[dim];
            }
            gates_raw[row] = sigmoid(acc);
        }
        // gates_raw[h*3+0] = g_cmp[h], gates_raw[h*3+1] = g_slc[h], gates_raw[h*3+2] = g_win[h]

        // ---- Per KV-head: compression attention probabilities ----
        // Needed for Eq. 9 importance aggregation (one p_cmp per query head,
        // but for group-sum in Eq. 10 we compute per query head).
        //
        // We store p_cmp for each (query_head, compress_block).
        // p_cmp[h][i] = probability that query head h assigns to compression block i,
        //               0 if block i is causally invalid for t.
        //
        // Causal validity: block i is valid iff i*d + l <= t+1
        //   (i.e. the LAST token of block i is at index i*d+l-1 <= t).

        let mut p_cmp_per_head = vec![vec![0.0_f32; n_cblocks]; num_heads];
        // Also store o_cmp per query head.
        let mut o_cmp_per_head = vec![vec![0.0_f32; head_dim]; num_heads];

        for h in 0..num_heads {
            let kv_h = h / n_rep;

            // Gather valid compressed keys for this query head
            let mut valid_indices: Vec<usize> = Vec::new();
            for i in 0..n_cblocks {
                if i * d + l <= t + 1 {
                    valid_indices.push(i);
                }
            }

            if valid_indices.is_empty() {
                // o_cmp stays zero; p_cmp stays zero
                continue;
            }

            // Query for compression branch: non-RoPE
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_t = &q_buf[q_off..q_off + head_dim];

            // Compute dot products with valid compressed keys
            let mut scores: Vec<f32> = valid_indices
                .iter()
                .map(|&i| {
                    let dot: f32 = q_t.iter().zip(ck[kv_h][i].iter()).map(|(a, b)| a * b).sum();
                    dot * scale
                })
                .collect();

            softmax_inplace(&mut scores);

            // Store probabilities back into full array (invalid positions stay 0)
            for (idx, &i) in valid_indices.iter().enumerate() {
                p_cmp_per_head[h][i] = scores[idx];
            }

            // o_cmp = Σ p * cv
            for (idx, &i) in valid_indices.iter().enumerate() {
                let w = scores[idx];
                for dim in 0..head_dim {
                    o_cmp_per_head[h][dim] += w * cv[kv_h][i][dim];
                }
            }
        }

        // ---- Eq. 9 + 10: per-KV-head importance scores for selection ----
        // Eq. 9 (paper-verbatim index, both m and n SUBTRACTED): importance of
        // selection block j for a single query head h:
        //   p_slc[h][j] = Σ_{m=0}^{cps-1} Σ_{n=0}^{ipc-1} p_cmp[h][cps*j - m - n]
        //
        // Eq. 10: group-sum across the n_rep query heads in kv_h's group:
        //   score_kv[kv_h][j] = Σ_{h in group(kv_h)} p_slc[h][j]
        //
        // Selection block j is causally valid for t iff j*l' < t+1
        // (at least one token of block j has been seen at or before t).

        let mut score_kv = vec![vec![0.0_f32; n_sblocks_total]; num_kv_heads];

        for kv_h in 0..num_kv_heads {
            // Group: query heads [kv_h * n_rep, (kv_h+1) * n_rep)
            let h_start = kv_h * n_rep;
            let h_end = h_start + n_rep;

            for j in 0..n_sblocks_total {
                // Eq. 9 (paper arXiv:2502.11089 §3.3.2): the importance of selection
                // block j for one query head sums the compression probabilities at
                // indices `p_cmp[cps*j - m - n]` (both m and n subtracted). Eq. 10 then
                // sums across the query heads in the KV group.
                let mut group_sum = 0.0_f32;
                let base = cps * j;
                for h in h_start..h_end {
                    let mut head_sum = 0.0_f32;
                    for m in 0..cps {
                        for n_idx in 0..ipc {
                            let offset = m + n_idx;
                            // base - offset; a negative index (small j) or one past the
                            // causal frontier carries no probability mass → skip.
                            if offset <= base {
                                let ci = base - offset;
                                if ci < n_cblocks {
                                    head_sum += p_cmp_per_head[h][ci];
                                }
                            }
                        }
                    }
                    group_sum += head_sum;
                }
                score_kv[kv_h][j] = group_sum;
            }
        }

        // ---- Top-n selection with forced blocks (ADR-042 §7) ----
        // Forced: block 0 (initial) + 2 local (highest-indexed valid).
        // "Valid" for selection: j*l' < t+1  →  j < (t+1).div_ceil(lp)
        //   equivalently: j*lp <= t  (since lp >= 1).
        // Remainder: top-(n-3) by score from non-forced valid blocks.
        // If fewer than n valid blocks exist, take all valid.

        // Per KV-head selection (all query heads in the group share the same blocks)
        let mut selected_per_kv: Vec<Vec<usize>> = vec![vec![]; num_kv_heads];

        for kv_h in 0..num_kv_heads {
            // Enumerate valid selection blocks for t
            let valid_sblocks: Vec<usize> =
                (0..n_sblocks_total).filter(|&j| j * lp < t + 1).collect();

            if valid_sblocks.is_empty() {
                // No valid selection blocks
                selected_per_kv[kv_h] = vec![];
                continue;
            }

            if valid_sblocks.len() <= n_sel {
                // Take all valid blocks
                selected_per_kv[kv_h] = valid_sblocks;
                continue;
            }

            // At least n_sel valid blocks — apply forced selection.
            // Forced candidates: block 0 and the 2 highest-indexed valid blocks.
            let mut forced: Vec<usize> = Vec::new();
            // Initial block
            forced.push(0);
            // 2 local blocks (highest-indexed valid, descending order)
            let last = valid_sblocks[valid_sblocks.len() - 1];
            let second_last = if valid_sblocks.len() >= 2 {
                valid_sblocks[valid_sblocks.len() - 2]
            } else {
                last
            };
            if !forced.contains(&last) {
                forced.push(last);
            }
            if !forced.contains(&second_last) {
                forced.push(second_last);
            }

            // Remaining free slots
            let n_free = n_sel.saturating_sub(forced.len());

            // Candidate pool: valid blocks NOT in forced
            let mut candidates: Vec<usize> = valid_sblocks
                .iter()
                .copied()
                .filter(|j| !forced.contains(j))
                .collect();

            // Sort candidates descending by score
            candidates.sort_by(|&a, &b| {
                score_kv[kv_h][b]
                    .partial_cmp(&score_kv[kv_h][a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_free: Vec<usize> = candidates.into_iter().take(n_free).collect();

            let mut sel: Vec<usize> = forced;
            sel.extend(top_free);
            sel.sort_unstable();
            sel.dedup();
            selected_per_kv[kv_h] = sel;
        }

        // ---- Selection branch output per query head ----
        let mut o_slc_per_head = vec![vec![0.0_f32; head_dim]; num_heads];

        for h in 0..num_heads {
            let kv_h = h / n_rep;
            let sel = &selected_per_kv[kv_h];

            if sel.is_empty() {
                continue;
            }

            // Gather token indices from selected blocks, with token-level causal mask.
            // Token `tok` in block `j` is causally valid for t iff tok <= t.
            let mut gathered_toks: Vec<usize> = Vec::new();
            for &j in sel {
                for pos in 0..lp {
                    let tok = j * lp + pos;
                    if tok < seq_len && tok <= t {
                        gathered_toks.push(tok);
                    }
                }
            }

            if gathered_toks.is_empty() {
                continue;
            }

            // Q (RoPE'd) for this head at position t
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_t = &q_rope_buf[q_off..q_off + head_dim];

            // Compute attention scores over gathered tokens
            let mut scores: Vec<f32> = gathered_toks
                .iter()
                .map(|&tok| {
                    let k_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                    let k_tok = &k_slc_buf[k_off..k_off + head_dim];
                    let dot: f32 = q_t.iter().zip(k_tok.iter()).map(|(a, b)| a * b).sum();
                    dot * scale
                })
                .collect();

            softmax_inplace(&mut scores);

            // Weighted sum of V
            for (idx, &tok) in gathered_toks.iter().enumerate() {
                let v_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                let v_tok = &v_slc_buf[v_off..v_off + head_dim];
                let w = scores[idx];
                for dim in 0..head_dim {
                    o_slc_per_head[h][dim] += w * v_tok[dim];
                }
            }
        }

        // ---- Sliding-window branch per query head ----
        let win_start = t.saturating_sub(win - 1);
        // Tokens [win_start, t] inclusive

        let mut o_win_per_head = vec![vec![0.0_f32; head_dim]; num_heads];

        for h in 0..num_heads {
            let kv_h = h / n_rep;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_t = &q_rope_buf[q_off..q_off + head_dim];

            let win_len = t - win_start + 1;
            let mut scores = vec![0.0_f32; win_len];
            for (si, tok) in (win_start..=t).enumerate() {
                let k_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                let k_tok = &k_win_buf[k_off..k_off + head_dim];
                let dot: f32 = q_t.iter().zip(k_tok.iter()).map(|(a, b)| a * b).sum();
                scores[si] = dot * scale;
            }
            softmax_inplace(&mut scores);

            for (si, tok) in (win_start..=t).enumerate() {
                let v_off = tok * num_kv_heads * head_dim + kv_h * head_dim;
                let v_tok = &v_win_buf[v_off..v_off + head_dim];
                let w = scores[si];
                for dim in 0..head_dim {
                    o_win_per_head[h][dim] += w * v_tok[dim];
                }
            }
        }

        // ---- Gated merge ----
        // o_t[h] = g_cmp[h] * o_cmp[h] + g_slc[h] * o_slc[h] + g_win[h] * o_win[h]
        for h in 0..num_heads {
            let g_cmp = gates_raw[h * 3];
            let g_slc = gates_raw[h * 3 + 1];
            let g_win = gates_raw[h * 3 + 2];
            let out_off = t * num_heads * head_dim + h * head_dim;
            for dim in 0..head_dim {
                output[out_off + dim] = g_cmp * o_cmp_per_head[h][dim]
                    + g_slc * o_slc_per_head[h][dim]
                    + g_win * o_win_per_head[h][dim];
            }
        }
    }

    output
}

// ===================================================================
// Test input builders
// ===================================================================

/// `(q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, weights)` — the 8 activation
/// buffers `apply_native_sparse_attention` takes, plus `x` and the learned weights.
type NsaInputs = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    NsaWeights,
);

/// Build a complete set of random NSA inputs for a given config and seq_len.
fn make_nsa_inputs(cfg: &NsaConfig, seq_len: usize, model_dim: usize, seed: u64) -> NsaInputs {
    let phi_in = cfg.phi_in();
    let head_dim = cfg.head_dim;

    // Independent K/V per branch (paper §3.3.3): 8 distinct activation buffers.
    let q = det_data(seq_len * cfg.q_dim(), seed);
    let q_rope = det_data(seq_len * cfg.q_dim(), seed + 1);
    let k_cmp = det_data(seq_len * cfg.kv_dim(), seed + 2);
    let k_slc = det_data(seq_len * cfg.kv_dim(), seed + 3);
    let k_win = det_data(seq_len * cfg.kv_dim(), seed + 4);
    let v_cmp = det_data(seq_len * cfg.kv_dim(), seed + 5);
    let v_slc = det_data(seq_len * cfg.kv_dim(), seed + 6);
    let v_win = det_data(seq_len * cfg.kv_dim(), seed + 7);
    let x = det_data(seq_len * model_dim, seed + 8);

    // Scale weights down to avoid extreme exp values.
    // φ weights: the MLP input is ~O(1); w1 shapes [phi_in, phi_in], large.
    // Scale phi_in by 1/phi_in keeps the first-layer pre-activations ~O(1).
    let phi_scale = 0.1 / (phi_in as f32).sqrt();
    let head_scale = 0.1 / (head_dim as f32).sqrt();
    let gate_scale = 0.1 / (model_dim as f32).sqrt();

    let scale_vec = |v: Vec<f32>, s: f32| v.into_iter().map(|x| x * s).collect::<Vec<_>>();

    let phi_k_w1 = scale_vec(det_data(phi_in * phi_in, seed + 10), phi_scale);
    let phi_k_b1 = scale_vec(det_data(phi_in, seed + 11), phi_scale);
    let phi_k_w2 = scale_vec(det_data(head_dim * phi_in, seed + 12), head_scale);
    let phi_k_b2 = scale_vec(det_data(head_dim, seed + 13), head_scale);

    let phi_v_w1 = scale_vec(det_data(phi_in * phi_in, seed + 14), phi_scale);
    let phi_v_b1 = scale_vec(det_data(phi_in, seed + 15), phi_scale);
    let phi_v_w2 = scale_vec(det_data(head_dim * phi_in, seed + 16), head_scale);
    let phi_v_b2 = scale_vec(det_data(head_dim, seed + 17), head_scale);

    let k_intrablock_pos = scale_vec(
        det_data(cfg.num_kv_heads * cfg.compress_block * head_dim, seed + 18),
        0.1,
    );
    let v_intrablock_pos = scale_vec(
        det_data(cfg.num_kv_heads * cfg.compress_block * head_dim, seed + 19),
        0.1,
    );

    let g_proj_w = scale_vec(
        det_data(3 * cfg.num_heads * model_dim, seed + 20),
        gate_scale,
    );
    let g_proj_b = scale_vec(det_data(3 * cfg.num_heads, seed + 21), 0.1);

    let weights = NsaWeights {
        phi_k_w1,
        phi_k_b1,
        phi_k_w2,
        phi_k_b2,
        phi_v_w1,
        phi_v_b1,
        phi_v_w2,
        phi_v_b2,
        k_intrablock_pos,
        v_intrablock_pos,
        g_proj_w,
        g_proj_b,
    };

    (
        q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, weights,
    )
}

// ===================================================================
// Integration tests
// ===================================================================

/// Parity test: kernel output matches naive oracle within 1e-4.
///
/// Covers MHA, GQA, early-token cases (seq_len < compress_block,
/// seq_len < select_block), and multi-block spanning sequences.
#[test]
fn test_nsa_parity_kernel_vs_oracle() {
    const TOLERANCE: f32 = 1e-4;

    // (seq_len, num_heads, num_kv_heads, head_dim,
    //  compress_block l, compress_stride d, select_block l', num_selected n, window w,
    //  model_dim, name)
    #[allow(clippy::type_complexity)]
    let cases: &[(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        &str,
    )] = &[
        // n >= 3 everywhere — the forced-block scheme (1 initial + 2 local) needs it,
        // and `NsaConfig::validate` now asserts it.
        // --- Early-token: seq_len < compress_block (no compression blocks) ---
        (1, 2, 2, 4, 4, 2, 4, 3, 8, 8, "early-no-cmp-seq1"),
        (3, 2, 2, 4, 4, 2, 4, 3, 8, 8, "early-no-cmp-seq3"),
        // --- Early-token: seq_len < select_block (no selection blocks) ---
        (3, 2, 2, 4, 4, 2, 8, 3, 8, 8, "early-no-slc-seq3"),
        // --- MHA: single block spanning ---
        (8, 2, 2, 4, 4, 2, 4, 3, 8, 8, "mha-2h-seq8-l4-d2-lp4"),
        // --- MHA: multiple blocks ---
        (12, 2, 2, 4, 4, 2, 4, 3, 8, 8, "mha-2h-seq12-l4-d2-lp4-n3"),
        // --- GQA: num_kv_heads < num_heads ---
        (8, 4, 2, 4, 4, 2, 4, 3, 8, 8, "gqa-4h-2kv-seq8"),
        (12, 4, 1, 4, 4, 2, 4, 3, 8, 8, "gqa-4h-1kv-seq12"),
        // --- Larger head_dim ---
        (10, 2, 2, 8, 4, 2, 4, 3, 8, 16, "mha-hd8-seq10"),
        // --- Window larger than seq_len (window-only effective) ---
        (6, 2, 2, 4, 4, 2, 4, 3, 32, 8, "win-large-seq6"),
        // --- Multiple compression blocks, GQA ---
        (16, 4, 2, 4, 4, 2, 4, 3, 8, 8, "gqa-4h-2kv-seq16-multi"),
        // --- Free-block path: n=5 with 6 valid selection blocks, so the
        //     importance-based top-(n-3) selection is actually exercised ---
        (24, 2, 2, 4, 4, 2, 4, 5, 8, 8, "free-path-seq24-n5"),
    ];

    for &(seq_len, num_heads, num_kv_heads, head_dim, l, d, lp, n_sel, win, model_dim, name) in
        cases
    {
        let cfg = NsaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            compress_block: l,
            compress_stride: d,
            select_block: lp,
            num_selected: n_sel,
            window: win,
        };

        let seed = (seq_len as u64) * 1009 + (num_heads as u64) * 31;
        let (q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, weights) =
            make_nsa_inputs(&cfg, seq_len, model_dim, seed);

        // Reference oracle
        let ref_out = ref_native_sparse_attention(
            &q, &q_rope, &k_cmp, &k_slc, &k_win, &v_cmp, &v_slc, &v_win, &x, &weights, seq_len,
            &cfg,
        );

        // Kernel under test
        let mut kernel_out = vec![0.0_f32; seq_len * cfg.q_dim()];
        let mut scratch = NsaScratch::default();
        apply_native_sparse_attention(
            &q,
            &q_rope,
            &k_cmp,
            &k_slc,
            &k_win,
            &v_cmp,
            &v_slc,
            &v_win,
            &x,
            &weights,
            &mut kernel_out,
            seq_len,
            &cfg,
            &mut scratch,
        );

        let diff = max_abs_diff(&kernel_out, &ref_out);
        assert!(
            diff <= TOLERANCE,
            "case '{name}': max_abs_diff={diff:.2e} exceeds tolerance {TOLERANCE:.1e}\n\
             This indicates a real discrepancy between kernel and oracle — do NOT loosen \
             the tolerance; investigate the disagreement."
        );
    }
}

/// Independence check: with window >= seq_len and gates forced window-only,
/// NSA output must equal `sigmoid(100) * dense_causal_attention(q_rope, k_win, v_win)`.
///
/// This test does NOT use the NSA oracle — it writes an independent dense attention
/// reference and checks a structural invariant of the gating mechanism.
#[test]
fn test_window_only_equals_dense_causal() {
    const TOLERANCE: f32 = 1e-4;

    let cfg = NsaConfig {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 64, // window >> seq_len → all tokens in window
    };
    let seq_len = 8;
    let model_dim = 8;
    let seed = 9999_u64;

    let (q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, _x, mut weights) =
        make_nsa_inputs(&cfg, seq_len, model_dim, seed);

    // Force gates: g_cmp ≈ 0, g_slc ≈ 0, g_win ≈ 1 via biases.
    // g_proj_w all zeros; g_proj_b = [-100, -100, +100] per head (row order: cmp, slc, win).
    weights.g_proj_w = vec![0.0_f32; 3 * cfg.num_heads * model_dim];
    weights.g_proj_b = Vec::with_capacity(3 * cfg.num_heads);
    for _h in 0..cfg.num_heads {
        weights.g_proj_b.push(-100.0); // g_cmp ≈ 0
        weights.g_proj_b.push(-100.0); // g_slc ≈ 0
        weights.g_proj_b.push(100.0); // g_win ≈ 1 = sigmoid(100)
    }

    // x can be anything since g_proj_w is zero; reuse a dummy
    let x = vec![0.0_f32; seq_len * model_dim];

    // Dense causal reference — the sliding-window branch (the only active branch under
    // these forced gates) uses q_rope / k_win / v_win, so the dense reference must too.
    let dense_out = ref_dense_causal_attention(
        &q_rope,
        &k_win,
        &v_win,
        seq_len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
    );

    let g_win = sigmoid(100.0_f32);
    let expected: Vec<f32> = dense_out.iter().map(|&x| g_win * x).collect();

    // NSA kernel
    let mut kernel_out = vec![0.0_f32; seq_len * cfg.q_dim()];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &q,
        &q_rope,
        &k_cmp,
        &k_slc,
        &k_win,
        &v_cmp,
        &v_slc,
        &v_win,
        &x,
        &weights,
        &mut kernel_out,
        seq_len,
        &cfg,
        &mut scratch,
    );

    let diff = max_abs_diff(&kernel_out, &expected);
    assert!(
        diff <= TOLERANCE,
        "window-only gate test: max_abs_diff={diff:.2e} exceeds {TOLERANCE:.1e}.\n\
         NSA with window >= seq_len and forced g_win=1 must match dense causal attention."
    );
}

/// Causal masking: position 0's output must not depend on future K/V.
///
/// Two runs differing ONLY in the six independent K/V branch buffers at positions
/// 1..seq_len. Position 0 output must be bit-identical. A later position must differ.
#[test]
fn test_causal_masking_integration() {
    let cfg = NsaConfig {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 8,
    };
    let seq_len = 8;
    let model_dim = 8;
    let seed = 7777_u64;

    let (q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, weights) =
        make_nsa_inputs(&cfg, seq_len, model_dim, seed);

    let kv_dim = cfg.kv_dim();

    // The six independent K/V branch buffers: k_cmp, k_slc, k_win, v_cmp, v_slc, v_win.
    let kv_base: [Vec<f32>; 6] = [k_cmp, k_slc, k_win, v_cmp, v_slc, v_win];

    // Perturb every K/V buffer at positions 1..seq_len.
    let kv_perturbed: [Vec<f32>; 6] = std::array::from_fn(|i| {
        let mut b = kv_base[i].clone();
        for pos in 1..seq_len {
            for d in 0..kv_dim {
                b[pos * kv_dim + d] += 12345.0;
            }
        }
        b
    });

    let run = |kv: &[Vec<f32>; 6]| {
        let mut out = vec![0.0_f32; seq_len * cfg.q_dim()];
        let mut scratch = NsaScratch::default();
        apply_native_sparse_attention(
            &q,
            &q_rope,
            &kv[0],
            &kv[1],
            &kv[2],
            &kv[3],
            &kv[4],
            &kv[5],
            &x,
            &weights,
            &mut out,
            seq_len,
            &cfg,
            &mut scratch,
        );
        out
    };

    let out_base = run(&kv_base);
    let out_perturbed = run(&kv_perturbed);

    let out_dim = cfg.q_dim();

    // Position 0: must be bit-identical (causally isolated)
    for d in 0..out_dim {
        assert_eq!(
            out_base[d].to_bits(),
            out_perturbed[d].to_bits(),
            "position 0 changed when only future K/V changed — causal mask leak at dim {d}"
        );
    }

    // Sanity: some later position must differ
    let later_changed = (1..seq_len).any(|pos| {
        (0..out_dim).any(|d| out_base[pos * out_dim + d] != out_perturbed[pos * out_dim + d])
    });
    assert!(
        later_changed,
        "no later position changed — the perturbation had no effect, which is unexpected"
    );
}

/// Selection-branch causal-leak regression (round-2 review, finding 1).
///
/// Soft-masking future tokens with a finite sentinel (e.g. `-10000`) leaks when a
/// *real* valid score falls below the sentinel: after softmax the masked future
/// token wins. Here the one causally-valid selection token (position 0) is given an
/// extreme-negative score, and the future tokens of position 0's own partial
/// selection block carry huge V values. With hard causal exclusion the future
/// tokens are never gathered, so position 0's output stays bounded near `v_slc[0]`.
#[test]
fn test_nsa_selection_no_future_leak() {
    let cfg = NsaConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 2,
        compress_block: 2,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 1,
    };
    let seq_len = 4;
    let model_dim = 4;
    let head_dim = cfg.head_dim;
    let kv_dim = cfg.kv_dim();

    // q_rope[0] · k_slc[0] is extreme-negative → the valid token's score is far
    // below any finite mask sentinel.
    let mut q_rope = vec![0.0_f32; seq_len * cfg.q_dim()];
    let mut k_slc = vec![0.0_f32; seq_len * kv_dim];
    for d in 0..head_dim {
        q_rope[d] = -1000.0; // query position 0
        k_slc[d] = 1000.0; // key position 0  → dot = -2e6
    }

    // v_slc: position 0 holds a small known value; positions 1..3 (future tokens
    // of position 0's own selection block [0, 4)) carry enormous values.
    let mut v_slc = vec![0.0_f32; seq_len * kv_dim];
    for d in 0..head_dim {
        v_slc[d] = 0.5; // position 0
    }
    for slot in v_slc.iter_mut().skip(kv_dim) {
        *slot = 1.0e9; // future tokens — must never leak into position 0
    }

    // Remaining buffers can be zero: compression has no causally-valid block for
    // qt=0, and the window (w=1) covers only position 0 itself.
    let q = vec![0.0_f32; seq_len * cfg.q_dim()];
    let k_cmp = vec![0.0_f32; seq_len * kv_dim];
    let k_win = vec![0.0_f32; seq_len * kv_dim];
    let v_cmp = vec![0.0_f32; seq_len * kv_dim];
    let v_win = vec![0.0_f32; seq_len * kv_dim];
    let x = vec![0.0_f32; seq_len * model_dim];

    // Correctly-shaped φ / gate weights, then force gates select-only:
    // g_cmp ≈ 0, g_slc ≈ 1, g_win ≈ 0 (row order [cmp, slc, win] for the 1 head).
    let (.., mut weights) = make_nsa_inputs(&cfg, seq_len, model_dim, 1);
    weights.g_proj_w = vec![0.0_f32; 3 * cfg.num_heads * model_dim];
    weights.g_proj_b = vec![-100.0, 100.0, -100.0];

    let mut out = vec![0.0_f32; seq_len * cfg.q_dim()];
    let mut scratch = NsaScratch::default();
    apply_native_sparse_attention(
        &q,
        &q_rope,
        &k_cmp,
        &k_slc,
        &k_win,
        &v_cmp,
        &v_slc,
        &v_win,
        &x,
        &weights,
        &mut out,
        seq_len,
        &cfg,
        &mut scratch,
    );

    // Position 0's selection branch gathers only token 0 (softmax of a single score
    // is 1.0), so o_slc[0] == v_slc[0] == 0.5. Compression has no valid block for
    // qt=0 and v_win is zero, so out[0] == g_slc * 0.5 == sigmoid(100) * 0.5 exactly.
    // Asserting the precise value (not just a loose bound) means a *disabled*
    // selection branch returning 0.0 fails the test too — not only a 1e9 leak.
    let expected = sigmoid(100.0_f32) * 0.5;
    for d in 0..head_dim {
        assert!(
            (out[d] - expected).abs() < 1.0e-4,
            "position 0 dim {d}: out={} expected≈{expected} — the selection branch \
             either leaked a future-token V (would be ~1e9) or returned 0.0",
            out[d]
        );
    }
}

/// Causal prefix-invariance regression (round-3 review, finding 1).
///
/// A causal prefill kernel must produce the same output for query `t` regardless of
/// how many tokens follow it. The selection-block count was computed with `floor`,
/// so a short prefix could have *zero* selection blocks while the same prefix inside
/// a longer sequence had one — silently activating the selection branch for earlier
/// queries. With `ceil` block counting the available block set depends only on `t`.
#[test]
fn test_nsa_prefix_invariance() {
    let cfg = NsaConfig {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        compress_block: 4,
        compress_stride: 2,
        select_block: 4,
        num_selected: 3,
        window: 4,
    };
    let long_len = 8;
    let short_len = 3; // < select_block: with `floor` this had 0 selection blocks
    let model_dim = 8;
    let seed = 4242_u64;

    // Build inputs for the long sequence; the short run uses the first `short_len`
    // tokens of every buffer — an exact prefix. Weights are seq_len-independent.
    let (q, q_rope, k_cmp, k_slc, k_win, v_cmp, v_slc, v_win, x, weights) =
        make_nsa_inputs(&cfg, long_len, model_dim, seed);

    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();

    let run = |len: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; len * q_dim];
        let mut scratch = NsaScratch::default();
        apply_native_sparse_attention(
            &q[..len * q_dim],
            &q_rope[..len * q_dim],
            &k_cmp[..len * kv_dim],
            &k_slc[..len * kv_dim],
            &k_win[..len * kv_dim],
            &v_cmp[..len * kv_dim],
            &v_slc[..len * kv_dim],
            &v_win[..len * kv_dim],
            &x[..len * model_dim],
            &weights,
            &mut out,
            len,
            &cfg,
            &mut scratch,
        );
        out
    };

    let out_long = run(long_len);
    let out_short = run(short_len);

    // The first `short_len` query outputs must be bit-identical: those queries see
    // exactly the same prefix in both runs.
    for i in 0..short_len * q_dim {
        assert_eq!(
            out_short[i].to_bits(),
            out_long[i].to_bits(),
            "prefix-invariance violated at output index {i} (query {}): \
             out_short={} out_long={} — appending future tokens changed an earlier output",
            i / q_dim,
            out_short[i],
            out_long[i]
        );
    }
}
