//! Native Sparse Attention (NSA) — Yuan et al., ACL 2025, arXiv:2502.11089.
//!
//! Three parallel causal branches merged via learned per-head gates:
//!
//! 1. **Compression** — overlapping KV blocks (length `l`, stride `d`) compressed by φ MLP;
//!    Q attends the compressed sequence for cheap global context (Eq. 7–8).
//! 2. **Selection** — importance scores (Eq. 9–10) identify top-`n` selection blocks (size `l'`);
//!    Q attends the gathered token-level KV at full resolution.
//! 3. **Sliding window** — standard causal local attention over the last `w` tokens.
//!
//! Output: `o_t = g_cmp·o_t^cmp + g_slc·o_t^slc + g_win·o_t^win` (Eq. 5).
//!
//! **Caller responsibility**: linear projections of Q/K/V and RoPE are the caller's
//! responsibility. Per paper §3.3.3 the three branches use **independent keys and values**
//! ("we provide independent keys and values for three branches" — to prevent shortcut
//! learning), so the kernel takes separately-projected `k_cmp`/`v_cmp`, `k_slc`/`v_slc`,
//! `k_win`/`v_win`. Compression uses **non-RoPE** Q/K (`q`, `k_cmp`) — its intra-block PE
//! already encodes within-block position; applying RoPE would double-count per
//! arXiv:2501.18795. Selection and sliding window use **RoPE'd** Q/K (`q_rope`, `k_slc`,
//! `k_win`). All V is RoPE-free. Queries are not independent per branch — the paper
//! specifies independent K/V only.
//!
//! Implementation is faithful to the paper's equations. Where `lucidrains` diverges from
//! the paper, the paper wins (see ADR-042 §Key Design Choice 1 for the full list).
//!
//! See ADR-042 for all design decisions.

use crate::forward::cpu::matmul_bt;

// ===================================================================
// Configuration
// ===================================================================

/// **Unstable**: configuration for one Native Sparse Attention layer.
///
/// Hyperparameter defaults from the paper §4.1: `l=32, d=16, l'=64, n=16, w=512`.
///
/// Invariants (asserted in [`NsaConfig::validate`] and [`apply_native_sparse_attention`]):
/// - `compress_block (l)` must be divisible by `compress_stride (d)`
/// - `select_block (l')` must be divisible by `compress_stride (d)`
/// - `compress_block (l) <= select_block (l')` — the paper's Eq. 9 precondition
/// - `num_heads % num_kv_heads == 0`
/// - `num_selected (n) >= 3` — the forced-block scheme needs 1 initial + 2 local blocks
/// - All other sizes > 0
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NsaConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of KV heads. `num_heads % num_kv_heads == 0` must hold.
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Compression block length `l`. Each compression block covers `l` consecutive tokens.
    pub compress_block: usize,
    /// Compression stride `d`. Consecutive compression blocks overlap by `l - d` tokens.
    /// `d | l` and `d | l'` must hold (Eq. 9 aggregation requires exact division).
    pub compress_stride: usize,
    /// Selection block length `l'`. Top-`n` selection blocks are attended at full resolution.
    pub select_block: usize,
    /// Number of top-scored selection blocks `n`. Must be `>= 3`: the scheme always
    /// includes a forced initial block + 2 local blocks, plus `n - 3` top-scored.
    pub num_selected: usize,
    /// Sliding window size `w`. Attention covers `k_rope[max(0,t-w+1)..=t]`.
    pub window: usize,
}

impl NsaConfig {
    /// **Unstable**: assert all divisibility and positivity invariants.
    ///
    /// Called at the top of [`apply_native_sparse_attention`] before any derived computation.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if any invariant is violated.
    pub fn validate(&self) {
        assert!(self.num_heads > 0, "num_heads must be > 0");
        assert!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        assert!(self.head_dim > 0, "head_dim must be > 0");
        assert!(self.compress_block > 0, "compress_block (l) must be > 0");
        assert!(self.compress_stride > 0, "compress_stride (d) must be > 0");
        assert!(self.select_block > 0, "select_block (l') must be > 0");
        // The paper's forced-block scheme is "1 initial + 2 local" blocks — undefined
        // for n < 3 (ADR-042 §KDC 7 already speaks of "the remaining n - 3").
        assert!(
            self.num_selected >= 3,
            "num_selected (n={}) must be >= 3 — the forced-block scheme needs \
             1 initial + 2 local blocks",
            self.num_selected
        );
        assert!(self.window > 0, "window (w) must be > 0");
        assert_eq!(
            self.num_heads % self.num_kv_heads,
            0,
            "num_heads ({}) must be divisible by num_kv_heads ({})",
            self.num_heads,
            self.num_kv_heads
        );
        // ADR-042 §Key Design Choice 5: d | l and d | l' required for Eq. 9 sum indexing.
        assert_eq!(
            self.compress_block % self.compress_stride,
            0,
            "compress_block (l={}) must be divisible by compress_stride (d={})",
            self.compress_block,
            self.compress_stride
        );
        assert_eq!(
            self.select_block % self.compress_stride,
            0,
            "select_block (l'={}) must be divisible by compress_stride (d={})",
            self.select_block,
            self.compress_stride
        );
        // ADR-042 §Key Design Choice 5: the paper scopes Eq. 9 to `l <= l'`.
        assert!(
            self.compress_block <= self.select_block,
            "compress_block (l={}) must be <= select_block (l'={}) — paper Eq. 9 precondition",
            self.compress_block,
            self.select_block
        );
    }

    /// **Unstable**: query heads per KV head (GQA repeat factor).
    #[inline]
    pub fn n_rep(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// **Unstable**: total Q buffer dimension per token: `num_heads * head_dim`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// **Unstable**: total KV buffer dimension per token: `num_kv_heads * head_dim`.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// **Unstable**: number of compression blocks for a sequence of `seq_len` tokens.
    ///
    /// Compression block `i` covers `[i*d, i*d + l)`. The last valid block `i` satisfies
    /// `i*d + l <= seq_len`, so there are `floor((seq_len - l) / d) + 1` blocks when
    /// `seq_len >= l`, and 0 otherwise.
    #[inline]
    pub fn num_compress_blocks(&self, seq_len: usize) -> usize {
        if seq_len < self.compress_block {
            return 0;
        }
        (seq_len - self.compress_block) / self.compress_stride + 1
    }

    /// **Unstable**: number of selection blocks for a sequence of `seq_len` tokens.
    ///
    /// Selection block `j` covers `[j*l', (j+1)*l')`. The count is `ceil(seq_len / l')`,
    /// **not** `floor` — a partial trailing block (one extending past `seq_len`) is still a
    /// real block. This is load-bearing for causal prefix-invariance: query `t`'s available
    /// selection-block set must depend only on `t`, never on the final `seq_len`. With a
    /// `floor` count, appending a future token could turn a previously-excluded partial
    /// block into a full one and retroactively change an earlier query's output. The
    /// non-existent / future tokens of a partial block are simply never gathered (the
    /// gather keeps only `tok <= qt`).
    #[inline]
    pub fn num_select_blocks(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.select_block)
    }

    /// **Unstable**: number of compression blocks that overlap one selection block = `l' / d`.
    ///
    /// Used in Eq. 9 (`p_cmp[cps*j - m - n]`): the importance of selection block `j`
    /// aggregates compression-block probabilities whose indices span
    /// `[cps*j - cps - ipc + 2, cps*j]`. See `aggregate_selection_importance`.
    #[inline]
    pub fn compress_per_select(&self) -> usize {
        self.select_block / self.compress_stride
    }

    /// **Unstable**: number of intra-block sub-strides per compression block = `l / d`.
    #[inline]
    pub fn intra_per_compress(&self) -> usize {
        self.compress_block / self.compress_stride
    }

    /// **Unstable**: φ MLP input dimension: `compress_block * head_dim`.
    #[inline]
    pub fn phi_in(&self) -> usize {
        self.compress_block * self.head_dim
    }
}

// ===================================================================
// Learned weights (caller-supplied, like DiffLambdaParams)
// ===================================================================

/// **Unstable**: caller-supplied learned NSA parameters for one layer.
///
/// Analogous to `DiffLambdaParams` in `differential.rs`. The kernel takes these as a struct
/// rather than owning them, so callers control weight storage and loading.
///
/// # Weight shapes (φ is shared across KV heads; ADR-042 §KDC 2)
///
/// - `phi_k_w1`: `[phi_in, phi_in]` row-major, where `phi_in = compress_block * head_dim`
/// - `phi_k_b1`: `[phi_in]`
/// - `phi_k_w2`: `[head_dim, phi_in]` row-major
/// - `phi_k_b2`: `[head_dim]`
/// - `phi_v_{w1,b1,w2,b2}`: identical shapes to φ_k, independent weights
/// - `k_intrablock_pos`: `[num_kv_heads * compress_block * head_dim]`
/// - `v_intrablock_pos`: same shape as `k_intrablock_pos`
/// - `g_proj_w`: `[3*num_heads, model_dim]` row-major
/// - `g_proj_b`: `[3*num_heads]`
///
/// Gate row ordering: `[h=0,cmp], [h=0,slc], [h=0,win], [h=1,cmp], ...`
pub struct NsaWeights {
    // φ_k: Linear(phi_in → phi_in) → ReLU → Linear(phi_in → head_dim)
    /// `[phi_in, phi_in]` stored row-major (`[out_row, in_col]`) for `matmul_bt`.
    pub phi_k_w1: Vec<f32>,
    /// `[phi_in]`
    pub phi_k_b1: Vec<f32>,
    /// `[head_dim, phi_in]` stored row-major for `matmul_bt`.
    pub phi_k_w2: Vec<f32>,
    /// `[head_dim]`
    pub phi_k_b2: Vec<f32>,

    // φ_v: same architecture as φ_k, independent weights
    pub phi_v_w1: Vec<f32>,
    pub phi_v_b1: Vec<f32>,
    pub phi_v_w2: Vec<f32>,
    pub phi_v_b2: Vec<f32>,

    /// Additive intra-block position encoding for K.
    /// Flat layout: `[kv_h * l * head_dim + pos_in_block * head_dim + dim]`.
    pub k_intrablock_pos: Vec<f32>,
    /// Additive intra-block position encoding for V. Same shape as `k_intrablock_pos`.
    pub v_intrablock_pos: Vec<f32>,

    /// Gate projection weight, `[3*num_heads, model_dim]`, row-major.
    pub g_proj_w: Vec<f32>,
    /// Gate projection bias, `[3*num_heads]`.
    pub g_proj_b: Vec<f32>,
}

// ===================================================================
// Scratch buffers
// ===================================================================

/// **Unstable**: pre-allocated scratch buffers for NSA; layout may grow.
#[derive(Default, Clone, Debug)]
pub struct NsaScratch {
    // --- φ MLP intermediates ---
    /// φ first-layer input (block flattened with PE): `[phi_in]`.
    phi_input: Vec<f32>,
    /// φ first-layer output (ReLU'd in place, then fed to layer 2): `[phi_in]`.
    phi_tmp1: Vec<f32>,

    // --- Compression branch ---
    /// Compressed K: `[num_kv_heads * max_cblocks * head_dim]`.
    ck: Vec<f32>,
    /// Compressed V: `[num_kv_heads * max_cblocks * head_dim]`.
    cv: Vec<f32>,
    /// Per-query compression scores (one KV head at a time): `[max_cblocks]`.
    compress_scores: Vec<f32>,
    /// Per-token compression branch output: `[seq_len * num_heads * head_dim]`.
    out_cmp: Vec<f32>,

    // --- Importance & selection ---
    /// Importance scores per KV head per selection block: `[num_kv_heads * max_sblocks]`.
    importance: Vec<f32>,
    /// Selected block indices per KV head: `[num_kv_heads * num_selected]`.
    sel_indices: Vec<usize>,
    /// Non-forced candidate block indices, reused across the per-token / per-KV-head
    /// top-`n` selection to avoid a hot-path allocation: capacity `max_sblocks`.
    sel_candidates: Vec<usize>,
    /// Gathered K for selection attention: `[n_valid_sel * l' * head_dim]`.
    sel_k: Vec<f32>,
    /// Gathered V for selection attention: same shape as `sel_k`.
    sel_v: Vec<f32>,
    /// Selection attention scores: `[n_valid_sel * l']`.
    sel_scores: Vec<f32>,
    /// Per-token selection branch output: `[seq_len * num_heads * head_dim]`.
    out_slc: Vec<f32>,

    // --- Sliding window branch ---
    /// Window attention scores: `[window]`.
    win_scores: Vec<f32>,
    /// Per-token sliding window branch output: `[seq_len * num_heads * head_dim]`.
    out_win: Vec<f32>,

    // --- Gate ---
    /// Gate values (post-sigmoid): `[3 * num_heads]` per token.
    gates: Vec<f32>,
}

impl NsaScratch {
    /// **Unstable**: resize all scratch buffers for the given sequence length and config.
    pub fn reserve_for(&mut self, seq_len: usize, cfg: &NsaConfig) {
        let phi_in = cfg.phi_in();
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let max_cblocks = cfg.num_compress_blocks(seq_len);
        let max_sblocks = cfg.num_select_blocks(seq_len);
        let n_sel = cfg.num_selected;
        let win = cfg.window;
        let lp = cfg.select_block;

        self.phi_input.resize(phi_in, 0.0);
        self.phi_tmp1.resize(phi_in, 0.0);
        // ck/cv are per-KV-head: num_kv_heads × max_cblocks × head_dim
        self.ck.resize(num_kv_heads * max_cblocks * head_dim, 0.0);
        self.cv.resize(num_kv_heads * max_cblocks * head_dim, 0.0);
        self.compress_scores.resize(max_cblocks, 0.0);
        self.out_cmp.resize(seq_len * num_heads * head_dim, 0.0);
        self.importance
            .resize(num_kv_heads * max_sblocks.max(1), 0.0);
        self.sel_indices.resize(num_kv_heads * n_sel, usize::MAX);
        self.sel_candidates.resize(max_sblocks, 0);
        // Each selected block is l' tokens; n_sel is the max selected count.
        self.sel_k.resize(n_sel * lp * head_dim, 0.0);
        self.sel_v.resize(n_sel * lp * head_dim, 0.0);
        self.sel_scores.resize(n_sel * lp, 0.0);
        self.out_slc.resize(seq_len * num_heads * head_dim, 0.0);
        self.win_scores.resize(win, 0.0);
        self.out_win.resize(seq_len * num_heads * head_dim, 0.0);
        self.gates.resize(3 * num_heads, 0.0);
    }
}

// ===================================================================
// Core kernel
// ===================================================================

/// **Unstable**: apply causal Native Sparse Attention (prefill, multi-token).
///
/// Implements the three-branch gated NSA algorithm from arXiv:2502.11089 §3.
///
/// # Buffer layouts
///
/// Per paper §3.3.3 the three branches use independent K/V; the kernel therefore takes
/// 8 caller-supplied activation buffers (ADR-042 §Key Design Choice 4):
///
/// - `q_buf`:      `[seq_len, num_heads * head_dim]` — non-RoPE query (compression)
/// - `q_rope_buf`: `[seq_len, num_heads * head_dim]` — RoPE'd query (selection, window)
/// - `k_cmp_buf`:  `[seq_len, num_kv_heads * head_dim]` — non-RoPE key, compression branch
/// - `k_slc_buf`:  `[seq_len, num_kv_heads * head_dim]` — RoPE'd key, selection branch
/// - `k_win_buf`:  `[seq_len, num_kv_heads * head_dim]` — RoPE'd key, sliding-window branch
/// - `v_cmp_buf`:  `[seq_len, num_kv_heads * head_dim]` — value, compression branch (RoPE-free)
/// - `v_slc_buf`:  `[seq_len, num_kv_heads * head_dim]` — value, selection branch (RoPE-free)
/// - `v_win_buf`:  `[seq_len, num_kv_heads * head_dim]` — value, sliding-window branch (RoPE-free)
/// - `x_buf`:      `[seq_len, model_dim]` — normed hidden state for gate
/// - `attn_out`:   `[seq_len, num_heads * head_dim]` — output
///
/// The compression branch uses non-RoPE Q/K (`q_buf`, `k_cmp_buf`); the selection and
/// sliding-window branches use RoPE'd Q/K (`q_rope_buf`, `k_slc_buf`, `k_win_buf`). V is
/// RoPE-free in all branches. Keys and values are independent *per branch* (separate
/// projections); queries are not — the paper specifies independent K/V only.
///
/// The model dimension `model_dim` is inferred from `x_buf.len() / seq_len`.
///
/// # Panics
///
/// Panics if any config invariant is violated or buffer lengths are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn apply_native_sparse_attention(
    q_buf: &[f32],
    q_rope_buf: &[f32],
    k_cmp_buf: &[f32],
    k_slc_buf: &[f32],
    k_win_buf: &[f32],
    v_cmp_buf: &[f32],
    v_slc_buf: &[f32],
    v_win_buf: &[f32],
    x_buf: &[f32],
    weights: &NsaWeights,
    attn_out: &mut [f32],
    seq_len: usize,
    cfg: &NsaConfig,
    scratch: &mut NsaScratch,
) {
    // Config validation must precede buffer-length checks: cfg.q_dim() etc. collapse to 0
    // when sizes are 0, making those checks vacuously pass; and a zero stride would reach
    // a divide-by-zero in num_compress_blocks before printing any useful message.
    cfg.validate();

    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let l = cfg.compress_block;
    let d = cfg.compress_stride;
    let lp = cfg.select_block;
    let n_sel = cfg.num_selected;
    let win = cfg.window;
    let n_rep = cfg.n_rep();
    let phi_in = cfg.phi_in();

    // Buffer-length asserts — real `assert!`, never `debug_assert!`.
    assert_eq!(
        q_buf.len(),
        seq_len * q_dim,
        "q_buf length mismatch: expected {}, got {}",
        seq_len * q_dim,
        q_buf.len()
    );
    assert_eq!(
        q_rope_buf.len(),
        seq_len * q_dim,
        "q_rope_buf length mismatch: expected {}, got {}",
        seq_len * q_dim,
        q_rope_buf.len()
    );
    assert_eq!(
        k_cmp_buf.len(),
        seq_len * kv_dim,
        "k_cmp_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        k_cmp_buf.len()
    );
    assert_eq!(
        k_slc_buf.len(),
        seq_len * kv_dim,
        "k_slc_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        k_slc_buf.len()
    );
    assert_eq!(
        k_win_buf.len(),
        seq_len * kv_dim,
        "k_win_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        k_win_buf.len()
    );
    assert_eq!(
        v_cmp_buf.len(),
        seq_len * kv_dim,
        "v_cmp_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        v_cmp_buf.len()
    );
    assert_eq!(
        v_slc_buf.len(),
        seq_len * kv_dim,
        "v_slc_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        v_slc_buf.len()
    );
    assert_eq!(
        v_win_buf.len(),
        seq_len * kv_dim,
        "v_win_buf length mismatch: expected {}, got {}",
        seq_len * kv_dim,
        v_win_buf.len()
    );
    assert_eq!(
        attn_out.len(),
        seq_len * q_dim,
        "attn_out length mismatch: expected {}, got {}",
        seq_len * q_dim,
        attn_out.len()
    );

    // model_dim: infer from x_buf length. seq_len=0 is handled below after the early return.
    // For seq_len>0 the division is safe because seq_len>=1.
    let model_dim = x_buf.len().checked_div(seq_len).unwrap_or(0);
    assert_eq!(
        x_buf.len(),
        seq_len * model_dim,
        "x_buf length must be seq_len*model_dim (model_dim inferred as {}), got {}",
        model_dim,
        x_buf.len()
    );
    // A non-empty sequence must carry a non-empty hidden state. Without this, an empty
    // x_buf infers model_dim = 0, the length check above passes vacuously, the g_proj_w
    // shape check is skipped, and the gate silently degrades to bias-only — masking a
    // caller that passed the wrong x_buf.
    assert!(
        seq_len == 0 || model_dim > 0,
        "x_buf must be non-empty when seq_len > 0 (model_dim inferred as 0)"
    );

    // Weight shape asserts — checked even for seq_len=0 to catch mis-constructed NsaWeights.
    // model_dim=0 when seq_len=0 (x_buf is empty), so g_proj_w check would expect 0 which
    // would pass vacuously — that's acceptable: the caller passed empty x_buf for empty seq.
    assert_eq!(
        weights.phi_k_w1.len(),
        phi_in * phi_in,
        "phi_k_w1 must be [phi_in, phi_in]=[{phi_in}*{phi_in}]"
    );
    assert_eq!(
        weights.phi_k_b1.len(),
        phi_in,
        "phi_k_b1 must be [phi_in={phi_in}]"
    );
    assert_eq!(
        weights.phi_k_w2.len(),
        head_dim * phi_in,
        "phi_k_w2 must be [head_dim={head_dim}, phi_in={phi_in}]"
    );
    assert_eq!(
        weights.phi_k_b2.len(),
        head_dim,
        "phi_k_b2 must be [head_dim={head_dim}]"
    );
    assert_eq!(
        weights.phi_v_w1.len(),
        phi_in * phi_in,
        "phi_v_w1 must be [phi_in, phi_in]=[{phi_in}*{phi_in}]"
    );
    assert_eq!(
        weights.phi_v_b1.len(),
        phi_in,
        "phi_v_b1 must be [phi_in={phi_in}]"
    );
    assert_eq!(
        weights.phi_v_w2.len(),
        head_dim * phi_in,
        "phi_v_w2 must be [head_dim={head_dim}, phi_in={phi_in}]"
    );
    assert_eq!(
        weights.phi_v_b2.len(),
        head_dim,
        "phi_v_b2 must be [head_dim={head_dim}]"
    );
    assert_eq!(
        weights.k_intrablock_pos.len(),
        num_kv_heads * l * head_dim,
        "k_intrablock_pos must be [num_kv_heads={num_kv_heads}, l={l}, head_dim={head_dim}]"
    );
    assert_eq!(
        weights.v_intrablock_pos.len(),
        num_kv_heads * l * head_dim,
        "v_intrablock_pos must be [num_kv_heads={num_kv_heads}, l={l}, head_dim={head_dim}]"
    );
    // g_proj_w shape depends on model_dim; skip the check when model_dim=0 (empty sequence).
    if model_dim > 0 {
        let g_proj_rows = 3 * num_heads;
        assert_eq!(
            weights.g_proj_w.len(),
            g_proj_rows * model_dim,
            "g_proj_w must be [3*num_heads={g_proj_rows}, model_dim={model_dim}]"
        );
    }
    assert_eq!(
        weights.g_proj_b.len(),
        3 * num_heads,
        "g_proj_b must be [3*num_heads={}]",
        3 * num_heads
    );

    if seq_len == 0 {
        return;
    }

    scratch.reserve_for(seq_len, cfg);
    attn_out.fill(0.0);
    scratch.out_cmp[..seq_len * q_dim].fill(0.0);
    scratch.out_slc[..seq_len * q_dim].fill(0.0);
    scratch.out_win[..seq_len * q_dim].fill(0.0);

    let scale = (head_dim as f32).powf(-0.5);
    let max_cblocks = cfg.num_compress_blocks(seq_len);
    let max_sblocks = cfg.num_select_blocks(seq_len);
    let cps = cfg.compress_per_select(); // l' / d
    let ipc = cfg.intra_per_compress(); // l / d

    // ================================================================
    // Step 1: Build compressed KV via φ MLP (Eq. 7)
    //
    // Compression block i covers tokens [i*d, i*d + l).
    // The intra-block position encoding (per KV head) is added to each token
    // in the block before feeding into φ.
    // φ: Linear(phi_in → phi_in) → ReLU → Linear(phi_in → head_dim)
    // φ weights are shared across KV heads; each KV head's blocks are compressed
    // independently but with the same φ weights.
    //
    // ck layout: [kv_h * max_cblocks * head_dim + bi * head_dim + dim]
    // ================================================================
    for kv_h in 0..num_kv_heads {
        let pos_enc_k = &weights.k_intrablock_pos[kv_h * l * head_dim..(kv_h + 1) * l * head_dim];
        let pos_enc_v = &weights.v_intrablock_pos[kv_h * l * head_dim..(kv_h + 1) * l * head_dim];

        for bi in 0..max_cblocks {
            let tok_start = bi * d;

            // ----- φ_k: apply to this block's K tokens -----
            // Build flattened block input with intra-block PE: phi_input = concat(k_tok + pe)
            let phi_in_buf = &mut scratch.phi_input[..phi_in];
            for p in 0..l {
                let tok = tok_start + p;
                let src = tok * kv_dim + kv_h * head_dim;
                let pe = p * head_dim;
                let dst = p * head_dim;
                for dd in 0..head_dim {
                    phi_in_buf[dst + dd] = k_cmp_buf[src + dd] + pos_enc_k[pe + dd];
                }
            }

            // Layer 1: tmp1 = ReLU(phi_input @ W1^T + b1)
            // matmul_bt(A[m,k], B[n,k], C[m,n]): C = A @ B^T
            // A = phi_input[1, phi_in], B = phi_k_w1[phi_in, phi_in], C[1, phi_in]
            // C[0,i] = phi_input · W1[i,:] = W1[i,:] · phi_input  ✓
            let tmp1 = &mut scratch.phi_tmp1[..phi_in];
            tmp1.fill(0.0);
            matmul_bt(phi_in_buf, &weights.phi_k_w1, tmp1, 1, phi_in, phi_in);
            for i in 0..phi_in {
                tmp1[i] = (tmp1[i] + weights.phi_k_b1[i]).max(0.0);
            }

            // Layer 2: ck[kv_h, bi] = tmp1 @ W2^T + b2
            // A = tmp1[1, phi_in], B = phi_k_w2[head_dim, phi_in], C[1, head_dim]
            let ck_off = (kv_h * max_cblocks + bi) * head_dim;
            let ck_slot = &mut scratch.ck[ck_off..ck_off + head_dim];
            ck_slot.fill(0.0);
            matmul_bt(tmp1, &weights.phi_k_w2, ck_slot, 1, phi_in, head_dim);
            for i in 0..head_dim {
                ck_slot[i] += weights.phi_k_b2[i];
            }

            // ----- φ_v: apply to this block's V tokens (independent weights) -----
            let phi_in_buf = &mut scratch.phi_input[..phi_in];
            for p in 0..l {
                let tok = tok_start + p;
                let src = tok * kv_dim + kv_h * head_dim;
                let pe = p * head_dim;
                let dst = p * head_dim;
                for dd in 0..head_dim {
                    phi_in_buf[dst + dd] = v_cmp_buf[src + dd] + pos_enc_v[pe + dd];
                }
            }
            let tmp1 = &mut scratch.phi_tmp1[..phi_in];
            tmp1.fill(0.0);
            matmul_bt(phi_in_buf, &weights.phi_v_w1, tmp1, 1, phi_in, phi_in);
            for i in 0..phi_in {
                tmp1[i] = (tmp1[i] + weights.phi_v_b1[i]).max(0.0);
            }
            let cv_off = (kv_h * max_cblocks + bi) * head_dim;
            let cv_slot = &mut scratch.cv[cv_off..cv_off + head_dim];
            cv_slot.fill(0.0);
            matmul_bt(tmp1, &weights.phi_v_w2, cv_slot, 1, phi_in, head_dim);
            for i in 0..head_dim {
                cv_slot[i] += weights.phi_v_b2[i];
            }
        }
    }

    // ================================================================
    // Per-token loop: process each query token qt in sequence order.
    // ================================================================
    for qt in 0..seq_len {
        // ----------------------------------------------------------------
        // 2a: Compression attention (Eq. 8) + Eq. 9 aggregation + Eq. 10 GQA sum
        //
        // Causal validity: compression block i is valid for query qt iff all of its
        // tokens are <= qt, i.e. i*d + l - 1 <= qt (equivalently i*d + l <= qt + 1).
        //
        // Importance is accumulated per KV head per selection block j by summing
        // the compression probabilities of all n_rep Q heads in that KV group
        // (Eq. 10), and for each Q head, the overlap sum from Eq. 9.
        // ----------------------------------------------------------------
        scratch.importance[..num_kv_heads * max_sblocks.max(1)].fill(0.0);

        for kv_h in 0..num_kv_heads {
            let valid_cblocks = count_valid_compress_blocks(qt, l, d, max_cblocks);

            let q_head_start = kv_h * n_rep;
            for qh_local in 0..n_rep {
                let qh = q_head_start + qh_local;

                // Compression branch output for this head (zero if no valid blocks).
                let out_cmp_off = qt * q_dim + qh * head_dim;
                let out_cmp_slot = &mut scratch.out_cmp[out_cmp_off..out_cmp_off + head_dim];

                if valid_cblocks == 0 {
                    // ADR-042 §KDC 10: empty branch → zero vector.
                    out_cmp_slot.fill(0.0);
                    continue;
                }

                // Score each valid compressed K: q · ck[i] * scale (non-RoPE Q, Eq. 8)
                let q_off = qt * q_dim + qh * head_dim;
                let q_head = &q_buf[q_off..q_off + head_dim];
                let scores = &mut scratch.compress_scores[..valid_cblocks];
                for bi in 0..valid_cblocks {
                    let ck_off = (kv_h * max_cblocks + bi) * head_dim;
                    let dot: f32 = q_head
                        .iter()
                        .zip(scratch.ck[ck_off..ck_off + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[bi] = dot * scale;
                }

                // Softmax → probabilities p_t^cmp (paper Eq. 8, softmax probs not logits)
                softmax_inplace(scores);

                // Compression branch output: o^cmp = Σ_i p_i * cv[kv_h, i]
                out_cmp_slot.fill(0.0);
                for bi in 0..valid_cblocks {
                    let p = scores[bi];
                    let cv_off = (kv_h * max_cblocks + bi) * head_dim;
                    for dd in 0..head_dim {
                        out_cmp_slot[dd] += p * scratch.cv[cv_off + dd];
                    }
                }

                // Eq. 9 (paper-verbatim index `(l'/d)*sj - m - n`, see
                // `aggregate_selection_importance`) + Eq. 10 (sum over Q-head group).
                let valid_sblocks = count_valid_select_blocks(qt, lp, max_sblocks);
                for sj in 0..valid_sblocks {
                    let block_score = aggregate_selection_importance(scores, sj, cps, ipc);
                    // Eq. 10: accumulate across Q heads in this KV group.
                    scratch.importance[kv_h * max_sblocks.max(1) + sj] += block_score;
                }
            }
        }

        // ----------------------------------------------------------------
        // 2b: Top-n selection per KV head (ADR-042 §KDC 7)
        //
        // Forced set: block 0 (initial) + up to 2 highest-indexed causally valid
        // blocks (local). Remaining (n - |forced|) slots go to top-scored blocks.
        // ----------------------------------------------------------------
        for kv_h in 0..num_kv_heads {
            let valid_sblocks = count_valid_select_blocks(qt, lp, max_sblocks);
            let sel_out = &mut scratch.sel_indices[kv_h * n_sel..(kv_h + 1) * n_sel];

            // Sentinel fill: usize::MAX means "not selected" in this slot.
            sel_out.fill(usize::MAX);

            if valid_sblocks == 0 {
                continue;
            }

            let n_take = n_sel.min(valid_sblocks);
            let imp = &scratch.importance
                [kv_h * max_sblocks.max(1)..kv_h * max_sblocks.max(1) + valid_sblocks];

            // Forced blocks: block 0 and up to 2 highest-indexed valid blocks.
            let mut forced = [usize::MAX; 3];
            // Initial block (always first).
            forced[0] = 0;
            let mut n_forced = 1usize;
            // Local block 1: highest-indexed valid block (distinct from 0 when valid_sblocks>1).
            if valid_sblocks >= 2 {
                forced[n_forced] = valid_sblocks - 1;
                n_forced += 1;
            }
            // Local block 2: second highest-indexed valid block.
            if valid_sblocks >= 3 {
                forced[n_forced] = valid_sblocks - 2;
                n_forced += 1;
            }
            // Sort for deterministic ordering, then clip to the n_take budget. The
            // forced entries are distinct by construction: block 0, and (when their
            // guards pass) valid_sblocks-1 and valid_sblocks-2, which are >= 1 and
            // distinct from each other and from 0.
            forced[..n_forced].sort_unstable();
            debug_assert!(
                forced[..n_forced].windows(2).all(|w| w[0] != w[1]),
                "forced selection blocks must be distinct by construction"
            );
            let n_forced = n_forced.min(n_take);

            // Collect non-forced blocks into scratch, pick top (n_take - n_forced) by
            // importance via an in-place partial selection sort (n_take is small, ~16).
            let n_extra = n_take - n_forced;
            let candidates = &mut scratch.sel_candidates;
            candidates.clear();
            for j in 0..valid_sblocks {
                if !forced[..n_forced].contains(&j) {
                    candidates.push(j);
                }
            }
            let n_extra = n_extra.min(candidates.len());
            for slot in 0..n_extra {
                let mut best = slot;
                for cand in (slot + 1)..candidates.len() {
                    if imp[candidates[cand]] > imp[candidates[best]] {
                        best = cand;
                    }
                }
                candidates.swap(slot, best);
            }

            // Write: forced blocks first, then top-extra.
            let mut out_idx = 0;
            for &fj in &forced[..n_forced] {
                if out_idx < n_take {
                    sel_out[out_idx] = fj;
                    out_idx += 1;
                }
            }
            for &ej in candidates[..n_extra].iter() {
                if out_idx < n_take {
                    sel_out[out_idx] = ej;
                    out_idx += 1;
                }
            }
            // Remaining slots stay at usize::MAX (sentinel).
        }

        // ----------------------------------------------------------------
        // 2c: Selection attention
        //
        // For each KV head: gather the selected blocks' K (RoPE'd) and V into
        // contiguous buffers — gathering *only* tokens `<= qt`. A selected block
        // may be the query's own, partial, block; its future tokens are never
        // gathered. This is hard causal exclusion, not a soft additive mask —
        // a soft mask leaks when a real score falls below the mask value.
        // ADR-042 §KDC 7. Then standard scaled-dot-product attention per Q head.
        // ----------------------------------------------------------------
        for kv_h in 0..num_kv_heads {
            let valid_sblocks = count_valid_select_blocks(qt, lp, max_sblocks);
            let sel_idxs = &scratch.sel_indices[kv_h * n_sel..(kv_h + 1) * n_sel];

            // Count valid (non-sentinel) selected indices.
            let n_valid_sel: usize = sel_idxs
                .iter()
                .filter(|&&i| i != usize::MAX && i < valid_sblocks)
                .count();

            if n_valid_sel == 0 {
                // No selection tokens — branch contributes zero for this KV-head group.
                continue;
            }

            // `n_valid_sel * lp` bounds the gathered count; scratch is sized for it.
            let max_sel_toks = n_valid_sel * lp;
            let sel_k_buf = &mut scratch.sel_k[..max_sel_toks * head_dim];
            let sel_v_buf = &mut scratch.sel_v[..max_sel_toks * head_dim];

            // Gather only causally-valid tokens (`tok <= qt`) — future tokens of a
            // partial selected block are never gathered.
            let mut n_gathered = 0usize;
            for &bj in sel_idxs.iter().take(n_sel) {
                if bj == usize::MAX || bj >= valid_sblocks {
                    continue;
                }
                let tok_start = bj * lp;
                for p in 0..lp {
                    let tok = tok_start + p;
                    // Hard causal exclusion. `tok > qt` also covers tokens past the end
                    // of the sequence in a partial trailing block: `qt < seq_len`, so
                    // `tok >= seq_len` implies `tok > qt`.
                    if tok > qt {
                        continue;
                    }
                    let dst = n_gathered * head_dim;
                    let src = tok * kv_dim + kv_h * head_dim;
                    sel_k_buf[dst..dst + head_dim].copy_from_slice(&k_slc_buf[src..src + head_dim]);
                    sel_v_buf[dst..dst + head_dim].copy_from_slice(&v_slc_buf[src..src + head_dim]);
                    n_gathered += 1;
                }
            }
            // Every valid selected block has its first token `bj*lp <= qt`, so
            // n_valid_sel > 0 implies at least one token was gathered.
            debug_assert!(n_gathered > 0, "n_valid_sel > 0 must imply n_gathered > 0");

            // Attend each Q head in this KV group against the gathered tokens.
            let q_head_start = kv_h * n_rep;
            for qh_local in 0..n_rep {
                let qh = q_head_start + qh_local;
                let q_off = qt * q_dim + qh * head_dim;
                let q_head = &q_rope_buf[q_off..q_off + head_dim];

                let scores = &mut scratch.sel_scores[..n_gathered];
                for ti in 0..n_gathered {
                    let k_off = ti * head_dim;
                    let dot: f32 = q_head
                        .iter()
                        .zip(sel_k_buf[k_off..k_off + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[ti] = dot * scale;
                }

                softmax_inplace(scores);

                let out_off = qt * q_dim + qh * head_dim;
                let out_slot = &mut scratch.out_slc[out_off..out_off + head_dim];
                out_slot.fill(0.0);
                for ti in 0..n_gathered {
                    let p = scores[ti];
                    let v_off = ti * head_dim;
                    for dd in 0..head_dim {
                        out_slot[dd] += p * sel_v_buf[v_off + dd];
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // 2d: Sliding window attention (ADR-042 §KDC 8)
        //
        // Attends k_rope[max(0, qt - w + 1) ..= qt] (inclusive).
        // Uses RoPE'd Q/K; V is RoPE-free.
        // ----------------------------------------------------------------
        let win_start = qt.saturating_sub(win - 1);
        let win_len = qt - win_start + 1; // ≥ 1 for all qt ≥ 0

        for kv_h in 0..num_kv_heads {
            let q_head_start = kv_h * n_rep;
            for qh_local in 0..n_rep {
                let qh = q_head_start + qh_local;
                let q_off = qt * q_dim + qh * head_dim;
                let q_head = &q_rope_buf[q_off..q_off + head_dim];

                let scores = &mut scratch.win_scores[..win_len];
                for (wi, tok) in (win_start..=qt).enumerate() {
                    let k_off = tok * kv_dim + kv_h * head_dim;
                    let dot: f32 = q_head
                        .iter()
                        .zip(k_win_buf[k_off..k_off + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[wi] = dot * scale;
                }

                softmax_inplace(scores);

                let out_off = qt * q_dim + qh * head_dim;
                let out_slot = &mut scratch.out_win[out_off..out_off + head_dim];
                out_slot.fill(0.0);
                for (wi, tok) in (win_start..=qt).enumerate() {
                    let p = scores[wi];
                    let v_off = tok * kv_dim + kv_h * head_dim;
                    for dd in 0..head_dim {
                        out_slot[dd] += p * v_win_buf[v_off + dd];
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // 2e: Gate computation (ADR-042 §KDC 9, Eq. 5)
        //
        // g_cmp, g_slc, g_win = sigmoid(W_g · x_t + b_g) per head.
        // W_g row layout: [h=0,cmp], [h=0,slc], [h=0,win], [h=1,cmp], ...
        // ----------------------------------------------------------------
        let x_t = &x_buf[qt * model_dim..(qt + 1) * model_dim];
        let gates = &mut scratch.gates[..3 * num_heads];
        for g_idx in 0..3 * num_heads {
            let w_row = &weights.g_proj_w[g_idx * model_dim..(g_idx + 1) * model_dim];
            let dot: f32 = x_t.iter().zip(w_row.iter()).map(|(&a, &b)| a * b).sum();
            gates[g_idx] = sigmoid(dot + weights.g_proj_b[g_idx]);
        }

        // ----------------------------------------------------------------
        // 2f: Gated merge (Eq. 5)
        //
        // o_t[qh] = g_cmp * o_cmp[qh] + g_slc * o_slc[qh] + g_win * o_win[qh]
        // Gates are independent sigmoids in [0,1], not normalized to sum to 1.
        // ----------------------------------------------------------------
        for qh in 0..num_heads {
            let g_cmp = gates[3 * qh];
            let g_slc = gates[3 * qh + 1];
            let g_win = gates[3 * qh + 2];

            let off = qt * q_dim + qh * head_dim;
            for dd in 0..head_dim {
                attn_out[off + dd] = g_cmp * scratch.out_cmp[off + dd]
                    + g_slc * scratch.out_slc[off + dd]
                    + g_win * scratch.out_win[off + dd];
            }
        }
    }
}

// ===================================================================
// Private helpers
// ===================================================================

/// Number of causally valid compression blocks for query position `qt`.
///
/// Block `i` covers tokens `[i*d, i*d+l)`; it is causally valid iff *all* its tokens are
/// `<= qt`, i.e. `i*d + l - 1 <= qt` (equivalently `i*d + l <= qt + 1`). A compressed block
/// is atomic — there is no token-level masking — so a partially-future block is excluded
/// entirely. See ADR-042 §KDC 5.
#[inline]
fn count_valid_compress_blocks(qt: usize, l: usize, d: usize, max_cblocks: usize) -> usize {
    if qt + 1 < l {
        return 0;
    }
    // max i s.t. i*d + l <= qt + 1  →  i <= (qt + 1 - l) / d
    ((qt + 1 - l) / d + 1).min(max_cblocks)
}

/// Number of causally valid selection blocks for query position `qt`.
///
/// Block `j` covers tokens `[j*l', (j+1)*l')`; it is causally valid iff it contains at
/// least one token `<= qt`, i.e. `j*l' <= qt` — so blocks `0..=qt/l'` are valid, giving
/// `qt/l' + 1`. A valid block may be **partial** (contain tokens `> qt`); the selection
/// branch gathers only its `<= qt` tokens (hard causal exclusion, not a soft mask).
///
/// The `.min(max_sblocks)` is a defensive upper bound that — given `num_select_blocks`
/// uses `ceil` — provably never binds: for any `qt < seq_len`, `qt/l' + 1 <=
/// ceil(seq_len/l') = max_sblocks`. So the result is `qt/l' + 1`, a function of `qt`
/// alone (causal prefix-invariance). See ADR-042 §KDC 7.
#[inline]
fn count_valid_select_blocks(qt: usize, lp: usize, max_sblocks: usize) -> usize {
    // max j s.t. j*l' <= qt  →  j <= qt / l'
    (qt / lp + 1).min(max_sblocks)
}

/// Eq. 9 selection-block importance: aggregate compression-branch probabilities.
///
/// For selection block `sj`, the paper's Eq. 9 (arXiv:2502.11089, §3.3.2) is:
///
/// ```text
/// p_t^slc[sj] = Σ_{m=0}^{cps-1} Σ_{n=0}^{ipc-1} p_t^cmp[cps*sj - m - n]
/// ```
///
/// where `cps = l'/d` and `ipc = l/d`. The compression index `cps*sj - m - n` has **both
/// `m` and `n` subtracted** (paper-verbatim — an earlier ADR draft had `+ m + n`, which is
/// wrong). `p_cmp` is the causally-valid compression-probability vector for query `t`; an
/// index outside `[0, p_cmp.len())` — negative for small `sj`, or beyond the causal
/// frontier — carries no probability mass and contributes 0. See ADR-042 §KDC 5 for the
/// worked index expansion.
#[inline]
fn aggregate_selection_importance(p_cmp: &[f32], sj: usize, cps: usize, ipc: usize) -> f32 {
    let base = cps * sj;
    let mut acc = 0.0_f32;
    for m in 0..cps {
        for n in 0..ipc {
            let offset = m + n;
            // `base - offset` underflows for small `sj`; a negative index is simply out of
            // range and contributes 0, same as an index past the causal frontier.
            if offset <= base {
                let ci = base - offset;
                if ci < p_cmp.len() {
                    acc += p_cmp[ci];
                }
            }
        }
    }
    acc
}

/// Numerically stable in-place softmax.
///
/// ADR-080 C1: routes the fail-closed decision through the shared row
/// finalizer. Previously `if sum > 0.0` had no `else` branch, so a NaN/+inf
/// score left `x` holding raw NaN/Inf exponentials instead of a zeroed row
/// (#780) — this is the single softmax helper shared by all three
/// `apply_native_sparse_attention` branches (compression, selection, sliding
/// window), so the fix applies to all of them at once.
#[inline]
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let (max_val, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(x);
    if crate::attention::softmax_row::row_fails_closed_pre_exp(max_val, any_nan) {
        x.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    crate::attention::softmax_row::finalize_row(x, sum);
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests;
