//! Reverse-mode backward (VJP) for a single GatedDeltaNet token sequence.
//!
//! Scope: computes grad_input (d_loss/d_input for each token) for a sequence
//! processed through one GDN layer, AND (when LoRA is present) the LoRA weight
//! gradients for each of the 5 GDN projections (qkv, z, b/beta, a/alpha, out).
//! This is "surface-B" LoRA: mirrors the GQA surface-A implementation in
//! backward/attention_gqa.rs, using the same lora_vjp primitive from ops.rs.
//!
//! # Forward recap (matches `gdn_fused.rs` hot path)
//!
//! For each timestep t (sequential, left to right):
//!
//! ```text
//! qkv_t   = W_qkv  @ x_t              (in_proj_qkv)
//! z_t     = W_z    @ x_t              (in_proj_z)
//! beta_t  = sigmoid(W_b @ x_t)        (in_proj_b)
//! alpha_t = W_a    @ x_t              (in_proj_a)
//!
//! c_t     = conv1d_silu(qkv_t)        (causal depthwise conv, then SiLU)
//!
//! For each value-head h  (key-head k_head = h / ratio):
//!   q_hat = L2norm( c_t[q_start..] )
//!   k_hat = L2norm( c_t[k_start..] )
//!   v     = c_t[v_start..]
//!
//!   g   = exp(-exp(a_log) * softplus(alpha_t[k_head] + dt_bias[k_head]))
//!
//!   kv_mem_t = S_{t-1}^T @ k_hat      (retrieval from *pre-decayed* state)
//!   delta_t  = (v - kv_mem_t * g) * beta_t
//!   S_t      = g * S_{t-1} + outer(k_hat, delta_t)   ← state update
//!
//!   o_t     = S_t^T @ q_hat * scale          (scale = 1/sqrt(key_dim))
//!
//! rms_out_t = gated_rms_norm(o_t, z_t, gamma)
//!           = (o_t / rms(o_t)) * gamma * SiLU(z_t)
//!
//! y_t = W_out @ rms_out_t                    (out_proj)
//! ```
//!
//! # Backward (reverse time, O(seq * heads * key_dim * value_dim))
//!
//! Run from t = T-1 down to 0.  Carry adjoint dS (same shape as S).
//! At each step, given grad_output dy_t = d_loss/d_y_t:
//!
//! ```text
//! d_rms_out  = W_out^T @ dy_t
//! → d_o_t, d_z_t  via gated_rms_norm backward
//! → d_q_hat, d_S  via  o = S^T q * scale  backward
//! → d_k_hat, d_delta, d_S_prev  via state-update + retrieval backward
//! → d_v, d_beta, d_kv_mem via delta backward
//! → d_alpha via g backward (chain through softplus, exp)
//! → d_q_raw, d_k_raw via L2-norm backward
//! → d_conv_out → d_qkv_t via conv1d SiLU backward
//! → d_x_t via W_qkv^T, W_z^T, W_b^T, W_a^T matmuls
//! ```
//!
//! State adjoint propagation:
//! ```text
//! dS_{t-1} = g * dS_t  +  (chain through rank-1 update and retrieval)
//! ```
//! Full derivation in inline comments per operation.
//!
//! # Numerical convention
//!
//! The gradcheck uses f64 accumulation in the finite-difference oracle to
//! eliminate catastrophic cancellation at eps = 1e-3.

use crate::attention::gdn::{sigmoid, softplus};
use crate::backward::ops::lora_vjp;
use crate::model::qwen35_config::Qwen35Config;

/// Gradients of the GDN layer w.r.t. LoRA params for all 5 projections.
///
/// Naming and layout mirror AttnGrads in backward/attention_gqa.rs.
/// grad_a_* shape: [rank * d_in], grad_b_* shape: [d_out * rank].
/// Fields are zero-length Vec when LoRA is absent (no-LoRA forward).
pub struct GdnGrads {
    /// in_proj_qkv LoRA: d_in=hidden, d_out=qkv_dim
    pub grad_a_qkv: Vec<f32>,
    pub grad_b_qkv: Vec<f32>,
    /// in_proj_z LoRA: d_in=hidden, d_out=output_dim
    pub grad_a_z: Vec<f32>,
    pub grad_b_z: Vec<f32>,
    /// in_proj_b (beta sigmoid) LoRA: d_in=hidden, d_out=num_kh
    pub grad_a_b: Vec<f32>,
    pub grad_b_b: Vec<f32>,
    /// in_proj_a (alpha decay) LoRA: d_in=hidden, d_out=num_kh
    pub grad_a_a: Vec<f32>,
    pub grad_b_a: Vec<f32>,
    /// out_proj LoRA: d_in=output_dim, d_out=hidden
    pub grad_a_out: Vec<f32>,
    pub grad_b_out: Vec<f32>,
    /// Input gradient (always present): [seq_len, hidden_size]
    pub dx: Vec<f32>,
}

/// Saved activations for one forward pass over a sequence.
///
/// All vectors are laid out as `[seq_len, ...]` row-major.  Per-head buffers
/// are further laid out as `[seq_len, value_heads, ...]`.
pub struct GdnSaved {
    pub seq_len: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub ratio: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub hidden_size: usize,
    pub qkv_dim: usize,
    pub output_dim: usize,
    pub kernel_size: usize,
    pub scale: f32,
    pub rms_eps: f32,

    /// Input tokens: [seq_len, hidden_size]
    pub inputs: Vec<f32>,

    /// Linear projections (pre-conv): [seq_len, qkv_dim]
    pub qkv_proj: Vec<f32>,
    /// z projection: [seq_len, output_dim]
    pub z_proj: Vec<f32>,
    /// raw beta pre-sigmoid: [seq_len, num_key_heads]
    pub beta_raw: Vec<f32>,
    /// alpha projection: [seq_len, num_key_heads]
    pub alpha_proj: Vec<f32>,
    /// beta = sigmoid(beta_raw): [seq_len, num_key_heads]
    pub beta: Vec<f32>,
    /// decay gate g per key-head: [seq_len, num_key_heads]
    pub g: Vec<f32>,

    /// Conv1d SiLU output: [seq_len, qkv_dim]
    pub conv_out: Vec<f32>,
    /// Conv rolling buffer states — the buffer content BEFORE each token is
    /// processed; needed to replay the conv backward.
    /// Shape: [seq_len, qkv_dim * (kernel_size - 1)]
    pub conv_buffers: Vec<f32>,

    /// L2-normalized q per value-head: [seq_len, value_heads, key_dim]
    pub q_hat: Vec<f32>,
    /// L2-normalized k per value-head: [seq_len, value_heads, key_dim]
    pub k_hat: Vec<f32>,
    /// v per value-head: [seq_len, value_heads, value_dim]
    pub v: Vec<f32>,
    /// L2 norm of raw q (pre-normalization): [seq_len, value_heads]
    pub q_norm: Vec<f32>,
    /// L2 norm of raw k: [seq_len, value_heads]
    pub k_norm: Vec<f32>,
    /// Exact forward denominator sqrt(||q||^2 + eps) used to normalize q: [seq_len, value_heads]
    pub q_eps_norm: Vec<f32>,
    /// Exact forward denominator sqrt(||k||^2 + eps) used to normalize k: [seq_len, value_heads]
    pub k_eps_norm: Vec<f32>,

    /// kv_mem = S_prev^T @ k_hat (retrieval before decay): [seq_len, value_heads, value_dim]
    pub kv_mem: Vec<f32>,

    /// Recurrent state S after each step: [seq_len, value_heads, key_dim * value_dim]
    /// S[t] is the state AFTER processing token t.
    /// S[-1] (initial) is implicitly zero.
    pub s_after: Vec<f32>,

    /// Per-head output o = S^T @ q * scale: [seq_len, value_heads, value_dim]
    pub o_heads: Vec<f32>,

    /// RMS norm value per head: [seq_len, value_heads]
    pub rms_vals: Vec<f32>,

    /// SiLU(z) per value-head: [seq_len, value_heads, value_dim]
    /// Stored because backward through gated_rms_norm needs it.
    pub silu_z: Vec<f32>,

    // ---- LoRA caches (populated only when LoRA is present) ----
    // h_* = A_proj @ x for each projection; shape [seq_len * lora_rank].
    // Empty when LoRA absent.
    /// LoRA rank used (0 = no LoRA)
    pub lora_rank: usize,
    /// LoRA scale factor
    pub lora_scale: f32,

    /// h_qkv = A_qkv @ x: [seq_len, lora_rank]
    pub h_qkv: Vec<f32>,
    /// h_z = A_z @ x: [seq_len, lora_rank]
    pub h_z: Vec<f32>,
    /// h_b = A_b @ x: [seq_len, lora_rank]  (beta sigmoid projection)
    pub h_b: Vec<f32>,
    /// h_a = A_a @ x: [seq_len, lora_rank]  (alpha decay projection)
    pub h_a: Vec<f32>,
    /// h_out = A_out @ gated_buf: [seq_len, lora_rank]  (output projection input)
    pub h_out: Vec<f32>,
    /// gated_buf (input to out_proj + LoRA_out): [seq_len, output_dim]
    /// Needed for the out_proj LoRA backward (x side of that linear).
    pub gated_buf: Vec<f32>,

    // ---- LoRA weight matrices (cloned from forward params for backward) ----
    // Storing these avoids threading 10 extra args through gdn_backward.
    // Empty Vec when LoRA absent.
    /// A_qkv: [rank, hidden]
    pub lora_a_qkv: Vec<f32>,
    /// B_qkv: [qkv_dim, rank]
    pub lora_b_qkv: Vec<f32>,
    /// A_z: [rank, hidden]
    pub lora_a_z: Vec<f32>,
    /// B_z: [output_dim, rank]
    pub lora_b_z: Vec<f32>,
    /// A_b: [rank, hidden]
    pub lora_a_b: Vec<f32>,
    /// B_b: [num_kh, rank]
    pub lora_b_b: Vec<f32>,
    /// A_a: [rank, hidden]
    pub lora_a_a: Vec<f32>,
    /// B_a: [num_kh, rank]
    pub lora_b_a: Vec<f32>,
    /// A_out: [rank, output_dim]
    pub lora_a_out: Vec<f32>,
    /// B_out: [hidden, rank]
    pub lora_b_out: Vec<f32>,
}

impl GdnSaved {
    /// Allocate a GdnSaved with all buffers zeroed.
    pub fn new(
        seq_len: usize,
        num_key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        hidden_size: usize,
        qkv_dim: usize,
        output_dim: usize,
        kernel_size: usize,
        scale: f32,
        rms_eps: f32,
    ) -> Self {
        let ratio = if num_key_heads == 0 {
            1
        } else {
            value_heads / num_key_heads
        };
        let buf_len = kernel_size.saturating_sub(1);
        Self {
            seq_len,
            num_key_heads,
            num_value_heads: value_heads,
            ratio,
            key_dim,
            value_dim,
            hidden_size,
            qkv_dim,
            output_dim,
            kernel_size,
            scale,
            rms_eps,
            inputs: vec![0.0; seq_len * hidden_size],
            qkv_proj: vec![0.0; seq_len * qkv_dim],
            z_proj: vec![0.0; seq_len * output_dim],
            beta_raw: vec![0.0; seq_len * num_key_heads],
            alpha_proj: vec![0.0; seq_len * num_key_heads],
            beta: vec![0.0; seq_len * num_key_heads],
            g: vec![0.0; seq_len * num_key_heads],
            conv_out: vec![0.0; seq_len * qkv_dim],
            conv_buffers: vec![0.0; seq_len * qkv_dim * buf_len],
            q_hat: vec![0.0; seq_len * value_heads * key_dim],
            k_hat: vec![0.0; seq_len * value_heads * key_dim],
            v: vec![0.0; seq_len * value_heads * value_dim],
            q_norm: vec![0.0; seq_len * value_heads],
            k_norm: vec![0.0; seq_len * value_heads],
            q_eps_norm: vec![0.0; seq_len * value_heads],
            k_eps_norm: vec![0.0; seq_len * value_heads],
            kv_mem: vec![0.0; seq_len * value_heads * value_dim],
            s_after: vec![0.0; seq_len * value_heads * key_dim * value_dim],
            o_heads: vec![0.0; seq_len * value_heads * value_dim],
            rms_vals: vec![0.0; seq_len * value_heads],
            silu_z: vec![0.0; seq_len * value_heads * value_dim],
            // LoRA caches: start empty; populated in gdn_forward_save when LoRA present.
            lora_rank: 0,
            lora_scale: 0.0,
            h_qkv: Vec::new(),
            h_z: Vec::new(),
            h_b: Vec::new(),
            h_a: Vec::new(),
            h_out: Vec::new(),
            gated_buf: Vec::new(),
            // LoRA weight matrices: start empty; cloned from params in gdn_forward_save.
            lora_a_qkv: Vec::new(),
            lora_b_qkv: Vec::new(),
            lora_a_z: Vec::new(),
            lora_b_z: Vec::new(),
            lora_a_b: Vec::new(),
            lora_b_b: Vec::new(),
            lora_a_a: Vec::new(),
            lora_b_a: Vec::new(),
            lora_a_out: Vec::new(),
            lora_b_out: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Forward pass that records all saved activations
// ---------------------------------------------------------------------------

/// Run the GDN forward for a full sequence, recording all saved activations
/// needed for the backward pass.
///
/// `inputs`:      [seq_len, hidden_size]
/// `weights`:     frozen GDN layer weights
/// `cfg`:         model config
/// `saved`:       output struct, must be pre-allocated via `GdnSaved::new`
/// `outputs`:     [seq_len, hidden_size] — written in-place
///
/// Optional LoRA parameters for the 5 GDN projections (qkv, z, b, a, out).
/// When all five pairs are `Some`, the forward adds `scale * (x @ A^T) @ B^T`
/// to each projection output and caches `h = A @ x` for the backward.
/// When any pair is `None`, the forward is byte-identical to the no-LoRA path.
#[allow(clippy::too_many_arguments)]
/// The ten LoRA weight-matrix slices bound for one GDN forward, in fixed order:
/// (a_qkv, b_qkv, a_z, b_z, a_b, b_b, a_a, b_a, a_out, b_out). Factored out to keep
/// the `lora_bound` binding under clippy's type-complexity threshold (this module is
/// `train-backward`-gated, so default clippy never lints it — see the app-bins gate).
type LoraBound<'a> = (
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
);

pub fn gdn_forward_save(
    inputs: &[f32],
    weights: &crate::attention::gdn::GatedDeltaNetWeights,
    _cfg: &Qwen35Config,
    saved: &mut GdnSaved,
    outputs: &mut [f32],
    // LoRA params for in_proj_qkv: a=[rank,hidden], b=[qkv_dim,rank]
    lora_a_qkv: Option<&[f32]>,
    lora_b_qkv: Option<&[f32]>,
    // LoRA params for in_proj_z: a=[rank,hidden], b=[output_dim,rank]
    lora_a_z: Option<&[f32]>,
    lora_b_z: Option<&[f32]>,
    // LoRA params for in_proj_b (beta): a=[rank,hidden], b=[num_kh,rank]
    lora_a_b: Option<&[f32]>,
    lora_b_b: Option<&[f32]>,
    // LoRA params for in_proj_a (alpha): a=[rank,hidden], b=[num_kh,rank]
    lora_a_a: Option<&[f32]>,
    lora_b_a: Option<&[f32]>,
    // LoRA params for out_proj: a=[rank,output_dim], b=[hidden,rank]
    lora_a_out: Option<&[f32]>,
    lora_b_out: Option<&[f32]>,
    lora_rank: usize,
    lora_scale: f32,
) {
    use crate::forward::cpu::matmul_bt;

    let seq_len = saved.seq_len;
    let hidden = saved.hidden_size;
    let num_kh = saved.num_key_heads;
    let value_heads = saved.num_value_heads;
    let ratio = saved.ratio;
    let key_dim = saved.key_dim;
    let value_dim = saved.value_dim;
    let qkv_dim = saved.qkv_dim;
    let output_dim = saved.output_dim;
    let kernel_size = saved.kernel_size;
    let scale = saved.scale;
    let rms_eps = saved.rms_eps;
    let buf_len = kernel_size.saturating_sub(1);
    let q_total = num_kh * key_dim;

    // Bind all ten LoRA slices at once.  If any pair is None or rank==0 the entire
    // LoRA path is skipped and the forward is byte-identical to the no-LoRA path.
    // Option<&[f32]> is Copy, so destructuring copies the refs — no heap allocation.
    let lora_bound: Option<LoraBound> = if lora_rank > 0 {
        match (
            lora_a_qkv, lora_b_qkv, lora_a_z, lora_b_z, lora_a_b, lora_b_b, lora_a_a, lora_b_a,
            lora_a_out, lora_b_out,
        ) {
            (
                Some(a_qkv),
                Some(b_qkv),
                Some(a_z),
                Some(b_z),
                Some(a_b),
                Some(b_b),
                Some(a_a),
                Some(b_a),
                Some(a_out),
                Some(b_out),
            ) => Some((a_qkv, b_qkv, a_z, b_z, a_b, b_b, a_a, b_a, a_out, b_out)),
            _ => None,
        }
    } else {
        None
    };

    // Reset LoRA cache state unconditionally. `saved` is a reusable buffer; a
    // prior LoRA-bound call would otherwise leave stale rank/matrices/h_* here.
    // gdn_backward gates its LoRA-gradient path on saved.lora_rank, so a leftover
    // nonzero rank after a no-LoRA call computes phantom gradients. The Some branch
    // below repopulates these; the None path now stays truly LoRA-free.
    saved.lora_rank = 0;
    saved.lora_scale = 0.0;
    saved.h_qkv.clear();
    saved.h_z.clear();
    saved.h_b.clear();
    saved.h_a.clear();
    saved.h_out.clear();
    saved.gated_buf.clear();
    saved.lora_a_qkv.clear();
    saved.lora_b_qkv.clear();
    saved.lora_a_z.clear();
    saved.lora_b_z.clear();
    saved.lora_a_b.clear();
    saved.lora_b_b.clear();
    saved.lora_a_a.clear();
    saved.lora_b_a.clear();
    saved.lora_a_out.clear();
    saved.lora_b_out.clear();

    // Initialise LoRA cache buffers on the saved struct when LoRA is active.
    // Also clone the weight matrices so gdn_backward doesn't need them threaded through.
    if let Some((a_qkv, b_qkv, a_z, b_z, a_b, b_b, a_a, b_a, a_out, b_out)) = lora_bound {
        saved.lora_rank = lora_rank;
        saved.lora_scale = lora_scale;
        saved.h_qkv = vec![0.0f32; seq_len * lora_rank];
        saved.h_z = vec![0.0f32; seq_len * lora_rank];
        saved.h_b = vec![0.0f32; seq_len * lora_rank];
        saved.h_a = vec![0.0f32; seq_len * lora_rank];
        saved.h_out = vec![0.0f32; seq_len * lora_rank];
        saved.gated_buf = vec![0.0f32; seq_len * output_dim];
        // Clone weight matrices for backward use.
        saved.lora_a_qkv = a_qkv.to_vec();
        saved.lora_b_qkv = b_qkv.to_vec();
        saved.lora_a_z = a_z.to_vec();
        saved.lora_b_z = b_z.to_vec();
        saved.lora_a_b = a_b.to_vec();
        saved.lora_b_b = b_b.to_vec();
        saved.lora_a_a = a_a.to_vec();
        saved.lora_b_a = b_a.to_vec();
        saved.lora_a_out = a_out.to_vec();
        saved.lora_b_out = b_out.to_vec();
    }

    // Rolling conv buffer (shared across time, updated per step)
    let mut conv_buf_live = vec![0.0f32; qkv_dim * buf_len];

    // Recurrent states per value-head
    let mut s_live = vec![0.0f32; value_heads * key_dim * value_dim];

    for t in 0..seq_len {
        let x = &inputs[t * hidden..(t + 1) * hidden];
        saved.inputs[t * hidden..(t + 1) * hidden].copy_from_slice(x);

        // --- Linear projections ---
        let qkv_out = &mut saved.qkv_proj[t * qkv_dim..(t + 1) * qkv_dim];
        matmul_bt(x, &weights.in_proj_qkv, qkv_out, 1, hidden, qkv_dim);

        // LoRA for in_proj_qkv: output += scale * (x @ A_qkv^T) @ B_qkv^T
        // Cache h_qkv = A_qkv @ x for the backward.
        if let Some((a_qkv, b_qkv, _, _, _, _, _, _, _, _)) = lora_bound {
            let h = &mut saved.h_qkv[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = a_qkv[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, xi)| a * xi)
                    .sum();
            }
            let qkv_out = &mut saved.qkv_proj[t * qkv_dim..(t + 1) * qkv_dim];
            for i in 0..qkv_dim {
                let acc: f32 = lora_scale
                    * b_qkv[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                qkv_out[i] += acc;
            }
        }

        let z_out = &mut saved.z_proj[t * output_dim..(t + 1) * output_dim];
        matmul_bt(x, &weights.in_proj_z, z_out, 1, hidden, output_dim);

        // LoRA for in_proj_z
        if let Some((_, _, a_z, b_z, _, _, _, _, _, _)) = lora_bound {
            let h = &mut saved.h_z[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = a_z[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, xi)| a * xi)
                    .sum();
            }
            let z_out = &mut saved.z_proj[t * output_dim..(t + 1) * output_dim];
            for i in 0..output_dim {
                let acc: f32 = lora_scale
                    * b_z[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                z_out[i] += acc;
            }
        }

        let beta_out = &mut saved.beta_raw[t * num_kh..(t + 1) * num_kh];
        matmul_bt(x, &weights.in_proj_b, beta_out, 1, hidden, num_kh);

        // LoRA for in_proj_b (beta, pre-sigmoid linear output)
        if let Some((_, _, _, _, a_b, b_b, _, _, _, _)) = lora_bound {
            let h = &mut saved.h_b[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = a_b[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, xi)| a * xi)
                    .sum();
            }
            let beta_out = &mut saved.beta_raw[t * num_kh..(t + 1) * num_kh];
            for i in 0..num_kh {
                let acc: f32 = lora_scale
                    * b_b[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                beta_out[i] += acc;
            }
        }

        let alpha_out = &mut saved.alpha_proj[t * num_kh..(t + 1) * num_kh];
        matmul_bt(x, &weights.in_proj_a, alpha_out, 1, hidden, num_kh);

        // LoRA for in_proj_a (alpha, pre-softplus linear output)
        if let Some((_, _, _, _, _, _, a_a, b_a, _, _)) = lora_bound {
            let h = &mut saved.h_a[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h[r] = a_a[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, xi)| a * xi)
                    .sum();
            }
            let alpha_out = &mut saved.alpha_proj[t * num_kh..(t + 1) * num_kh];
            for i in 0..num_kh {
                let acc: f32 = lora_scale
                    * b_a[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                alpha_out[i] += acc;
            }
        }

        // sigmoid(beta)
        for kh in 0..num_kh {
            let raw = saved.beta_raw[t * num_kh + kh];
            saved.beta[t * num_kh + kh] = sigmoid(raw);
        }

        // decay gate g per key-head
        for kh in 0..num_kh {
            let alpha_h = saved.alpha_proj[t * num_kh + kh];
            let a = weights.a_log[kh].exp();
            let sp = softplus(alpha_h + weights.dt_bias[kh]);
            saved.g[t * num_kh + kh] = (-a * sp).exp();
        }

        // --- Causal conv1d + SiLU ---
        // Save conv buffer state BEFORE this step
        let cb_off = t * qkv_dim * buf_len;
        saved.conv_buffers[cb_off..cb_off + qkv_dim * buf_len].copy_from_slice(&conv_buf_live);

        let conv_out_t = &mut saved.conv_out[t * qkv_dim..(t + 1) * qkv_dim];
        let qkv_in = &saved.qkv_proj[t * qkv_dim..(t + 1) * qkv_dim];
        conv1d_silu_fwd(
            qkv_in,
            &mut conv_buf_live,
            &weights.conv1d_weight,
            conv_out_t,
            qkv_dim,
            kernel_size,
        );

        // --- Per value-head recurrence ---
        for h in 0..value_heads {
            let kh = h / ratio;
            let q_start = kh * key_dim;
            let k_start = q_total + kh * key_dim;
            let v_start = q_total * 2 + h * value_dim;

            // Raw q, k, v from conv output
            let mut q_raw =
                saved.conv_out[t * qkv_dim + q_start..t * qkv_dim + q_start + key_dim].to_vec();
            let mut k_raw =
                saved.conv_out[t * qkv_dim + k_start..t * qkv_dim + k_start + key_dim].to_vec();
            let v_slice = &saved.conv_out[t * qkv_dim + v_start..t * qkv_dim + v_start + value_dim];

            // Save v
            let v_off = (t * value_heads + h) * value_dim;
            saved.v[v_off..v_off + value_dim].copy_from_slice(v_slice);

            // L2-normalize q, k and save norms
            let q_sum_sq = l2_norm_sq(&q_raw);
            let k_sum_sq = l2_norm_sq(&k_raw);
            let q_norm = q_sum_sq.sqrt().max(1e-6_f32.sqrt());
            let k_norm = k_sum_sq.sqrt().max(1e-6_f32.sqrt());
            saved.q_norm[t * value_heads + h] = q_norm;
            saved.k_norm[t * value_heads + h] = k_norm;
            // Save the exact eps-stabilised denominator used for normalization.
            // These differ from q_norm/k_norm when sum_sq < 1e-6 and must be
            // used verbatim in the backward to avoid a ~29% gradient error near zero.
            let q_eps_norm = (q_sum_sq + 1e-6).sqrt();
            let k_eps_norm = (k_sum_sq + 1e-6).sqrt();
            saved.q_eps_norm[t * value_heads + h] = q_eps_norm;
            saved.k_eps_norm[t * value_heads + h] = k_eps_norm;
            for v in &mut q_raw {
                *v /= q_eps_norm;
            }
            for v in &mut k_raw {
                *v /= k_eps_norm;
            }
            let q_off = (t * value_heads + h) * key_dim;
            let k_off = (t * value_heads + h) * key_dim;
            saved.q_hat[q_off..q_off + key_dim].copy_from_slice(&q_raw);
            saved.k_hat[k_off..k_off + key_dim].copy_from_slice(&k_raw);

            let g_h = saved.g[t * num_kh + kh];
            let beta_h = saved.beta[t * num_kh + kh];

            // S_{t-1} for this head
            let s_off = h * key_dim * value_dim;
            let s = &s_live[s_off..s_off + key_dim * value_dim];

            // kv_mem = S_{t-1}^T @ k_hat
            let kvm_off = (t * value_heads + h) * value_dim;
            let kv_mem_h = &mut saved.kv_mem[kvm_off..kvm_off + value_dim];
            kv_mem_h.fill(0.0);
            for i in 0..key_dim {
                let ki = k_raw[i];
                for j in 0..value_dim {
                    kv_mem_h[j] += s[i * value_dim + j] * ki;
                }
            }

            // delta = (v - kv_mem * g) * beta
            let mut delta = vec![0.0f32; value_dim];
            for j in 0..value_dim {
                delta[j] = (v_slice[j] - kv_mem_h[j] * g_h) * beta_h;
            }

            // S_t = g * S_{t-1} + outer(k_hat, delta)
            let s_mut = &mut s_live[s_off..s_off + key_dim * value_dim];
            for i in 0..key_dim {
                for j in 0..value_dim {
                    s_mut[i * value_dim + j] = s_mut[i * value_dim + j] * g_h + k_raw[i] * delta[j];
                }
            }

            // Save S_t
            let sa_off = (t * value_heads + h) * key_dim * value_dim;
            saved.s_after[sa_off..sa_off + key_dim * value_dim].copy_from_slice(s_mut);

            // o = S_t^T @ q_hat * scale
            let o_off = (t * value_heads + h) * value_dim;
            let o_h = &mut saved.o_heads[o_off..o_off + value_dim];
            o_h.fill(0.0);
            for i in 0..key_dim {
                let qi = q_raw[i];
                for j in 0..value_dim {
                    o_h[j] += s_mut[i * value_dim + j] * qi;
                }
            }
            for j in 0..value_dim {
                o_h[j] *= scale;
            }
        }

        // --- Gated RMSNorm + output projection ---
        let z_slice = &saved.z_proj[t * output_dim..(t + 1) * output_dim];
        let gamma = &weights.norm_weight[..value_dim];
        let mut gated_buf = vec![0.0f32; output_dim];

        for h in 0..value_heads {
            let o_off = (t * value_heads + h) * value_dim;
            let o_h = &saved.o_heads[o_off..o_off + value_dim];
            let z_h = &z_slice[h * value_dim..(h + 1) * value_dim];

            // Compute and save RMS value
            let sum_sq: f32 = o_h.iter().map(|v| v * v).sum();
            let rms = (sum_sq / value_dim as f32 + rms_eps).sqrt();
            saved.rms_vals[t * value_heads + h] = rms;
            let inv_rms = 1.0 / rms;

            // Compute and save SiLU(z)
            let sz_off = (t * value_heads + h) * value_dim;
            for j in 0..value_dim {
                let sz = silu_f32(z_h[j]);
                saved.silu_z[sz_off + j] = sz;
                gated_buf[h * value_dim + j] = (o_h[j] * inv_rms) * gamma[j] * sz;
            }
        }

        // Save gated_buf when LoRA is active (needed as the x-side input for out_proj LoRA backward).
        if lora_bound.is_some() {
            saved.gated_buf[t * output_dim..(t + 1) * output_dim].copy_from_slice(&gated_buf);
        }

        let y_t = &mut outputs[t * hidden..(t + 1) * hidden];
        matmul_bt(&gated_buf, &weights.out_proj, y_t, 1, output_dim, hidden);

        // LoRA for out_proj: y_t += scale * (gated_buf @ A_out^T) @ B_out^T
        // out_proj is [hidden, output_dim], so d_out=hidden, d_in=output_dim.
        // Cache h_out = A_out @ gated_buf for the backward.
        if let Some((_, _, _, _, _, _, _, _, a_out, b_out)) = lora_bound {
            let h_slot = &mut saved.h_out[t * lora_rank..(t + 1) * lora_rank];
            for r in 0..lora_rank {
                h_slot[r] = a_out[r * output_dim..(r + 1) * output_dim]
                    .iter()
                    .zip(gated_buf.iter())
                    .map(|(a, xi)| a * xi)
                    .sum();
            }
            let y_t = &mut outputs[t * hidden..(t + 1) * hidden];
            for i in 0..hidden {
                let acc: f32 = lora_scale
                    * b_out[i * lora_rank..(i + 1) * lora_rank]
                        .iter()
                        .zip(h_slot.iter())
                        .map(|(b, hi)| b * hi)
                        .sum::<f32>();
                y_t[i] += acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

/// VJP of the GDN sequence forward.
///
/// Returns a `GdnGrads` containing the input gradient (dx, always populated)
/// and — when the saved activations carry LoRA caches (i.e. `gdn_forward_save`
/// was called with LoRA params) — the weight gradients for each of the 5
/// projections. When LoRA is absent, the grad_a_*/grad_b_* Vecs are empty.
///
/// `grad_outputs`:  [seq_len, hidden_size] — upstream gradient of loss w.r.t. output
/// `saved`:         activations from `gdn_forward_save`
/// `weights`:       same frozen weights used in forward
pub fn gdn_backward(
    grad_outputs: &[f32],
    saved: &GdnSaved,
    weights: &crate::attention::gdn::GatedDeltaNetWeights,
) -> GdnGrads {
    let seq_len = saved.seq_len;
    let hidden = saved.hidden_size;
    let num_kh = saved.num_key_heads;
    let value_heads = saved.num_value_heads;
    let ratio = saved.ratio;
    let key_dim = saved.key_dim;
    let value_dim = saved.value_dim;
    let qkv_dim = saved.qkv_dim;
    let output_dim = saved.output_dim;
    let kernel_size = saved.kernel_size;
    let scale = saved.scale;
    let buf_len = kernel_size.saturating_sub(1);
    let q_total = num_kh * key_dim;
    let gamma = &weights.norm_weight[..value_dim];

    let have_lora = saved.lora_rank > 0 && !saved.h_qkv.is_empty();
    let lora_rank = saved.lora_rank;
    let lora_scale = saved.lora_scale;

    // Allocate LoRA weight grad accumulators (empty when no LoRA).
    let mut grad_a_qkv = if have_lora {
        vec![0.0f32; lora_rank * hidden]
    } else {
        Vec::new()
    };
    let mut grad_b_qkv = if have_lora {
        vec![0.0f32; qkv_dim * lora_rank]
    } else {
        Vec::new()
    };
    let mut grad_a_z = if have_lora {
        vec![0.0f32; lora_rank * hidden]
    } else {
        Vec::new()
    };
    let mut grad_b_z = if have_lora {
        vec![0.0f32; output_dim * lora_rank]
    } else {
        Vec::new()
    };
    let mut grad_a_b = if have_lora {
        vec![0.0f32; lora_rank * hidden]
    } else {
        Vec::new()
    };
    let mut grad_b_b = if have_lora {
        vec![0.0f32; num_kh * lora_rank]
    } else {
        Vec::new()
    };
    let mut grad_a_a = if have_lora {
        vec![0.0f32; lora_rank * hidden]
    } else {
        Vec::new()
    };
    let mut grad_b_a = if have_lora {
        vec![0.0f32; num_kh * lora_rank]
    } else {
        Vec::new()
    };
    let mut grad_a_out = if have_lora {
        vec![0.0f32; lora_rank * output_dim]
    } else {
        Vec::new()
    };
    let mut grad_b_out = if have_lora {
        vec![0.0f32; hidden * lora_rank]
    } else {
        Vec::new()
    };

    let mut grad_inputs = vec![0.0f32; seq_len * hidden];

    // Adjoint state dS: accumulated across timesteps, flows backward.
    // dS[h] is dL/d(S_t) for head h, updated as t decreases.
    let mut d_s = vec![0.0f32; value_heads * key_dim * value_dim];

    // Gradient accumulator for qkv_proj across all timesteps.
    //
    // The causal conv1d has a rolling buffer: at time t the buffer slot tb
    // holds qkv_proj[t - (buf_len - tb)].  So the grad of loss w.r.t.
    // qkv_proj[t] has contributions not only from the conv at step t (via
    // W_conv[ch, buf_len]) but also from the conv at steps t+1 … t+buf_len
    // (via W_conv[ch, 0..buf_len-1]).  We collect all contributions into a
    // full-sequence buffer so that when we apply W_qkv^T at step t the
    // accumulator already holds the complete gradient for that token.
    let mut d_qkv_proj_all = vec![0.0f32; seq_len * qkv_dim];

    for t in (0..seq_len).rev() {
        let dy = &grad_outputs[t * hidden..(t + 1) * hidden];
        let x_t = &saved.inputs[t * hidden..(t + 1) * hidden];

        // ---- 1. Backward through out_proj: d_gated_buf = W_out^T @ dy ----
        // W_out is [hidden, output_dim].  out = gated_buf @ W_out^T means
        // d_gated_buf[j] = sum_i W_out[i,j] * dy[i]
        //
        // When LoRA is present: y_t = base_out + scale*(gated_buf@A_out^T)@B_out^T
        // The upstream gradient dy is w.r.t. the total y_t.
        // lora_vjp(dy, gated_buf_t, h_out_t, a_out, b_out, rank, output_dim, hidden, scale)
        // returns (grad_b_out delta, grad_a_out delta, d_gated_buf_lora_delta).
        // d_gated_buf = W_out^T dy  +  lora_vjp dx contribution (d_gated_buf from LoRA path).
        let mut d_gated_buf = vec![0.0f32; output_dim];
        for j in 0..output_dim {
            let mut acc = 0.0f64;
            for i in 0..hidden {
                acc += weights.out_proj[i * output_dim + j] as f64 * dy[i] as f64;
            }
            d_gated_buf[j] = acc as f32;
        }

        // LoRA weight grads for out_proj + dx contribution to d_gated_buf.
        // out_proj layout: [hidden, output_dim] → d_in=output_dim, d_out=hidden.
        // lora_vjp(g=dy, x=gated_buf_t, h=h_out_t, a=lora_a_out, b=lora_b_out,
        //          rank, d_in=output_dim, d_out=hidden, scale)
        if have_lora {
            let gated_buf_t = &saved.gated_buf[t * output_dim..(t + 1) * output_dim];
            let h_out_t = &saved.h_out[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                dy,
                gated_buf_t,
                h_out_t,
                &saved.lora_a_out,
                &saved.lora_b_out,
                lora_rank,
                output_dim,
                hidden,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_out[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_out[k] += gb[k];
            }
            for j in 0..output_dim {
                d_gated_buf[j] += dx_lora[j];
            }
        }

        // ---- 2. Backward through gated RMSNorm for each value-head ----
        let z_slice = &saved.z_proj[t * output_dim..(t + 1) * output_dim];
        let mut d_o_heads = vec![0.0f32; output_dim];
        let mut d_z_proj = vec![0.0f32; output_dim];

        for h in 0..value_heads {
            let rms = saved.rms_vals[t * value_heads + h];
            let inv_rms = 1.0 / rms;
            let o_off = (t * value_heads + h) * value_dim;
            let o_h = &saved.o_heads[o_off..o_off + value_dim];
            let sz_off = (t * value_heads + h) * value_dim;
            let silu_z_h = &saved.silu_z[sz_off..sz_off + value_dim];

            // Forward: gated[j] = (o[j] * inv_rms) * gamma[j] * silu_z[j]
            // i.e.  gated[j] = x_norm[j] * gamma[j] * silu_z[j]
            //       where x_norm[j] = o[j] / rms

            let d_g = &d_gated_buf[h * value_dim..(h + 1) * value_dim];
            let d_o = &mut d_o_heads[h * value_dim..(h + 1) * value_dim];
            let d_z = &mut d_z_proj[h * value_dim..(h + 1) * value_dim];

            // d_xnorm[j] = d_g[j] * gamma[j] * silu_z[j]
            // d_silu_z[j] = d_g[j] * gamma[j] * x_norm[j]
            let mut d_xnorm = vec![0.0f32; value_dim];
            for j in 0..value_dim {
                let x_norm_j = o_h[j] * inv_rms;
                d_xnorm[j] = d_g[j] * gamma[j] * silu_z_h[j];
                // d_silu_z then backward through SiLU
                let d_silu_z_j = d_g[j] * gamma[j] * x_norm_j;
                let z_j = z_slice[h * value_dim + j];
                // SiLU(z) = z * sigmoid(z)
                // d/dz SiLU(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
                //               = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
                let sig_z = sigmoid(z_j);
                let d_silu_dz = sig_z * (1.0 + z_j * (1.0 - sig_z));
                d_z[j] = d_silu_z_j * d_silu_dz;
            }

            // RMSNorm backward:
            // x_norm[j] = o[j] / rms,  rms = sqrt(mean(o^2) + eps)
            // d_o[j] = (d_xnorm[j] / rms) - (o[j] / (rms^3 * dim)) * sum_k d_xnorm[k] * o[k]
            let dot_dxnorm_o: f32 = d_xnorm.iter().zip(o_h.iter()).map(|(a, b)| a * b).sum();
            let rms3_dim = rms * rms * rms * value_dim as f32;
            for j in 0..value_dim {
                d_o[j] = d_xnorm[j] * inv_rms - o_h[j] * dot_dxnorm_o / rms3_dim;
            }
        }

        // ---- 3. Accumulate d_z_proj → d_x via W_z^T, then LoRA weight grads ----
        // z_proj = W_z @ x  (+LoRA when present),  d_x += W_z^T @ d_z_proj
        //
        // d_z_proj IS the pre-nonlinearity linear-projection-output gradient for z.
        // (z goes through SiLU in the gated RMSNorm; d_z_proj is the gradient
        // w.r.t. the linear output of in_proj_z, BEFORE SiLU — that is what step 2
        // computes via the SiLU backward inside the RMSNorm loop.)
        let dx_t = &mut grad_inputs[t * hidden..(t + 1) * hidden];
        for j in 0..output_dim {
            let dz_j = d_z_proj[j];
            if dz_j.abs() < f32::EPSILON {
                continue;
            }
            for i in 0..hidden {
                dx_t[i] += weights.in_proj_z[j * hidden + i] * dz_j;
            }
        }
        // LoRA weight grads for in_proj_z.
        // g = d_z_proj, x = x_t, h = h_z_t, d_in=hidden, d_out=output_dim
        if have_lora {
            let h_z_t = &saved.h_z[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                &d_z_proj,
                x_t,
                h_z_t,
                &saved.lora_a_z,
                &saved.lora_b_z,
                lora_rank,
                hidden,
                output_dim,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_z[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_z[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }
        }

        // ---- 4. Per value-head recurrence backward ----
        // We process heads in REVERSE order (arbitrary — no inter-head deps).
        // We work with the per-head adjoint dS[h] which carries across time.

        let mut d_conv_out = vec![0.0f32; qkv_dim];
        // Per key-head accumulators for alpha and beta scalar grads (multiple
        // value-heads share one key-head, so we must sum across value-heads first
        // before applying lora_vjp to avoid double-counting the d_in/d_out shape).
        let mut d_alpha_kh = vec![0.0f32; num_kh]; // d_loss / d(alpha_proj[t, kh])
        let mut d_beta_raw_kh = vec![0.0f32; num_kh]; // d_loss / d(beta_raw[t, kh])

        for h in 0..value_heads {
            let kh = h / ratio;
            let q_start = kh * key_dim;
            let k_start = q_total + kh * key_dim;
            let v_start = q_total * 2 + h * value_dim;

            let g_h = saved.g[t * num_kh + kh];
            let beta_h = saved.beta[t * num_kh + kh];

            let q_off = (t * value_heads + h) * key_dim;
            let k_off = (t * value_heads + h) * key_dim;
            let kvm_off = (t * value_heads + h) * value_dim;
            let sa_off = (t * value_heads + h) * key_dim * value_dim;
            let ds_off = h * key_dim * value_dim;

            let q_hat_h = &saved.q_hat[q_off..q_off + key_dim];
            let k_hat_h = &saved.k_hat[k_off..k_off + key_dim];
            let kv_mem_h = &saved.kv_mem[kvm_off..kvm_off + value_dim];
            let s_t = &saved.s_after[sa_off..sa_off + key_dim * value_dim];
            let d_o_h = &d_o_heads[h * value_dim..(h + 1) * value_dim];
            let d_s_h = &mut d_s[ds_off..ds_off + key_dim * value_dim];

            // ---- 4a. Backward through o = S_t^T @ q * scale ----
            // o[j] = scale * sum_i S_t[i,j] * q[i]
            // dS_t[i,j] += scale * d_o[j] * q[i]   (accumulate into existing dS_h from t+1)
            // d_q[i]     = scale * sum_j S_t[i,j] * d_o[j]
            let mut d_q = vec![0.0f32; key_dim];
            for i in 0..key_dim {
                let mut acc = 0.0f32;
                for j in 0..value_dim {
                    d_s_h[i * value_dim + j] += scale * d_o_h[j] * q_hat_h[i];
                    acc += s_t[i * value_dim + j] * d_o_h[j];
                }
                d_q[i] = scale * acc;
            }

            // ---- 4b. Backward through S_t = g * S_{t-1} + outer(k, delta) ----
            //
            // S_t[i,j] = g * S_{t-1}[i,j] + k[i] * delta[j]
            //
            // dS_{t-1}[i,j] = g * dS_t[i,j]
            // d_g           = sum_{i,j} S_{t-1}[i,j] * dS_t[i,j]
            // d_k[i]        = sum_j dS_t[i,j] * delta[j]
            // d_delta[j]    = sum_i dS_t[i,j] * k[i]
            //
            // S_{t-1} is either s_after[t-1] or zero (t=0).

            // Reconstruct delta from saved activations:
            // delta[j] = (v[j] - kv_mem[j] * g) * beta
            let v_off_h = (t * value_heads + h) * value_dim;
            let v_h = &saved.v[v_off_h..v_off_h + value_dim];
            let mut delta = vec![0.0f32; value_dim];
            for j in 0..value_dim {
                delta[j] = (v_h[j] - kv_mem_h[j] * g_h) * beta_h;
            }

            // d_g from state update and retrieval (see below)
            // First: from state update S_t = g * S_{t-1} + outer(k, delta)
            let s_prev_slice: Vec<f32> = if t == 0 {
                vec![0.0f32; key_dim * value_dim]
            } else {
                let prev_off = ((t - 1) * value_heads + h) * key_dim * value_dim;
                saved.s_after[prev_off..prev_off + key_dim * value_dim].to_vec()
            };

            let mut d_g_h: f32 = 0.0;
            let mut d_k = vec![0.0f32; key_dim];
            let mut d_delta = vec![0.0f32; value_dim];

            for i in 0..key_dim {
                for j in 0..value_dim {
                    let ds_ij = d_s_h[i * value_dim + j];
                    d_g_h += s_prev_slice[i * value_dim + j] * ds_ij;
                    d_k[i] += ds_ij * delta[j];
                    d_delta[j] += ds_ij * k_hat_h[i];
                }
            }

            // Propagate dS_{t-1}: dS_h now holds d_S_{t-1}
            for ij in 0..key_dim * value_dim {
                d_s_h[ij] *= g_h;
            }

            // ---- 4c. Backward through kv_mem = S_{t-1}^T @ k and delta computation ----
            //
            // delta[j] = (v[j] - kv_mem[j] * g) * beta
            // d_v[j]       = d_delta[j] * beta
            // d_kv_mem[j]  = -d_delta[j] * g * beta  (chain through -kv_mem*g)
            //   → but d_kv_mem = d_delta * (-g * beta) for each j
            // d_g       += sum_j (-kv_mem[j] * beta) * d_delta[j]
            // d_beta    = sum_j (v[j] - kv_mem[j] * g) * d_delta[j]  = sum_j delta[j]/beta * d_delta[j]
            //           = (1/beta) * dot(delta, d_delta)

            let mut d_v = vec![0.0f32; value_dim];
            let mut d_kv_mem = vec![0.0f32; value_dim];
            let mut d_beta_h: f32 = 0.0;

            for j in 0..value_dim {
                d_v[j] = d_delta[j] * beta_h;
                d_kv_mem[j] = -d_delta[j] * g_h * beta_h;
                d_g_h += (-kv_mem_h[j] * beta_h) * d_delta[j];
                // d_beta += d(delta)/d(beta) * d_delta[j]
                //         = (v[j] - kv_mem[j]*g) * d_delta[j]
                // Computed directly from base without dividing by beta, which
                // suppresses gradients for saturated (near-zero) beta values.
                d_beta_h += (v_h[j] - kv_mem_h[j] * g_h) * d_delta[j];
            }

            // ---- 4d. Backward through kv_mem = S_{t-1}^T @ k ----
            // kv_mem[j] = sum_i S_{t-1}[i,j] * k[i]
            // dS_{t-1}[i,j] += d_kv_mem[j] * k[i]
            // d_k[i]         += sum_j S_{t-1}[i,j] * d_kv_mem[j]
            for i in 0..key_dim {
                for j in 0..value_dim {
                    d_s_h[i * value_dim + j] += d_kv_mem[j] * k_hat_h[i];
                    d_k[i] += s_prev_slice[i * value_dim + j] * d_kv_mem[j];
                }
            }

            // ---- 4e. Backward through g (decay gate) ----
            // g = exp(-exp(a_log) * softplus(alpha + dt_bias))
            // dg/d_alpha = g * (-exp(a_log)) * sigmoid(alpha + dt_bias)
            //            = g * (-exp(a_log)) * softplus'(alpha + dt_bias)
            // where softplus'(x) = sigmoid(x)
            let alpha_h = saved.alpha_proj[t * num_kh + kh];
            let a_val = weights.a_log[kh].exp();
            let sp_arg = alpha_h + weights.dt_bias[kh];
            let sp_deriv = sigmoid(sp_arg); // d/dx softplus(x) = sigmoid(x)
            // dg/d_alpha = g * (-a_val) * sp_deriv
            let d_alpha_from_g = d_g_h * g_h * (-a_val) * sp_deriv;

            // ---- 4f. Backward through L2-norm of q and k ----
            // q_hat = q_raw / ||q_raw||_2   (eps-stabilised)
            // d_q_raw[i] = (d_q[i] / norm) - q_hat[i] * dot(q_hat, d_q) / norm
            //            = (d_q[i] - q_hat[i] * dot(q_hat, d_q)) / norm
            // Same for k.

            let q_eps_norm = saved.q_eps_norm[t * value_heads + h];
            let k_eps_norm = saved.k_eps_norm[t * value_heads + h];

            let dot_q = dot(&d_q, q_hat_h);
            let dot_k = dot(&d_k, k_hat_h);
            let mut d_q_raw = vec![0.0f32; key_dim];
            let mut d_k_raw = vec![0.0f32; key_dim];
            for i in 0..key_dim {
                d_q_raw[i] = (d_q[i] - q_hat_h[i] * dot_q) / q_eps_norm;
                d_k_raw[i] = (d_k[i] - k_hat_h[i] * dot_k) / k_eps_norm;
            }

            // ---- 4g. Accumulate into d_conv_out ----
            for i in 0..key_dim {
                d_conv_out[q_start + i] += d_q_raw[i];
                d_conv_out[k_start + i] += d_k_raw[i];
            }
            for j in 0..value_dim {
                d_conv_out[v_start + j] += d_v[j];
            }

            // ---- 4h. Backward through sigmoid(beta_raw) → d_beta_raw ----
            // beta = sigmoid(beta_raw)
            // d_beta_raw = d_beta * beta * (1 - beta)
            let beta_raw_h = saved.beta_raw[t * num_kh + kh];
            let sig = sigmoid(beta_raw_h);
            let d_beta_raw_h = d_beta_h * sig * (1.0 - sig);

            // ---- 4i. Accumulate per key-head scalar grads ----
            // Multiple value-heads share the same key-head; accumulate into kh slots.
            // These will be fed to W_a^T/W_b^T and lora_vjp after this loop.
            d_alpha_kh[kh] += d_alpha_from_g;
            d_beta_raw_kh[kh] += d_beta_raw_h;
        } // end per-head loop

        // ---- Apply per key-head alpha/beta grads → d_x and LoRA weight grads ----
        // d_alpha_proj[t, kh] = d_alpha_kh[kh]  (scalar per head)
        // d_beta_raw[t, kh]   = d_beta_raw_kh[kh]
        //
        // For lora_vjp on the beta and alpha projections, we need the full
        // d_proj_output vector of shape [num_kh].  We built those above.
        //
        // g for in_proj_b  = d_beta_raw_kh   (pre-sigmoid linear-output gradient)
        // g for in_proj_a  = d_alpha_kh      (pre-softplus linear-output gradient)
        let dx_t = &mut grad_inputs[t * hidden..(t + 1) * hidden];
        for kh in 0..num_kh {
            let d_a = d_alpha_kh[kh];
            let d_b = d_beta_raw_kh[kh];
            for i in 0..hidden {
                dx_t[i] += weights.in_proj_a[kh * hidden + i] * d_a;
                dx_t[i] += weights.in_proj_b[kh * hidden + i] * d_b;
            }
        }
        // LoRA weight grads for in_proj_a and in_proj_b.
        // g = d_alpha_kh (full [num_kh] vector), x = x_t, h_a_t, d_in=hidden, d_out=num_kh
        if have_lora {
            let h_b_t = &saved.h_b[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                &d_beta_raw_kh,
                x_t,
                h_b_t,
                &saved.lora_a_b,
                &saved.lora_b_b,
                lora_rank,
                hidden,
                num_kh,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_b[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_b[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }

            let h_a_t = &saved.h_a[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                &d_alpha_kh,
                x_t,
                h_a_t,
                &saved.lora_a_a,
                &saved.lora_b_a,
                lora_rank,
                hidden,
                num_kh,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_a[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_a[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }
        }

        // ---- 5. Backward through conv1d + SiLU ----
        //
        // Forward: sum_pre[ch] = sum_{tb} buf[ch,tb] * w[ch,tb] + qkv[t,ch] * w[ch,buf_len]
        //          c[t,ch]     = SiLU(sum_pre[ch])
        //
        // buf[ch, tb] = qkv_proj[t - (buf_len - tb), ch]   (for t - (buf_len-tb) >= 0)
        //
        // So d_qkv_proj[t][ch] gets contributions from:
        //   - conv at step t:  d_sum * w[ch, buf_len]
        //   - conv at step t': d_sum(t') * w[ch, tb]  where t' > t and t = t' - (buf_len - tb)
        //
        // We write the current-step contribution into d_qkv_proj_all[t] and the
        // buffer-slot contributions into d_qkv_proj_all[src_t] for src_t < t.
        // Because we iterate backward, src_t < t has not yet had its W_qkv^T applied
        // (step 6 runs later in the outer loop), so those writes land correctly.

        let cb_off = t * qkv_dim * buf_len;
        let conv_buf_t = &saved.conv_buffers[cb_off..cb_off + qkv_dim * buf_len];
        let qkv_in_t = &saved.qkv_proj[t * qkv_dim..(t + 1) * qkv_dim];

        for ch in 0..qkv_dim {
            let w_off = ch * kernel_size;
            let buf_off = ch * buf_len;

            // Recompute sum_pre (same as forward)
            let mut sum_pre = 0.0f32;
            for tb in 0..buf_len {
                sum_pre += conv_buf_t[buf_off + tb] * weights.conv1d_weight[w_off + tb];
            }
            sum_pre += qkv_in_t[ch] * weights.conv1d_weight[w_off + buf_len];

            let sig = sigmoid(sum_pre);
            let silu_deriv = sig * (1.0 + sum_pre * (1.0 - sig));
            let d_sum = d_conv_out[ch] * silu_deriv;

            // Current-timestep contribution (through qkv_proj[t])
            d_qkv_proj_all[t * qkv_dim + ch] += d_sum * weights.conv1d_weight[w_off + buf_len];

            // Buffer-slot contributions: buf[ch, tb] came from qkv_proj[src_t]
            // where src_t = t - (buf_len - tb).
            for tb in 0..buf_len {
                let lag = buf_len - tb; // 1-based distance into the past
                if t >= lag {
                    let src_t = t - lag;
                    d_qkv_proj_all[src_t * qkv_dim + ch] +=
                        d_sum * weights.conv1d_weight[w_off + tb];
                }
            }
        }

        // ---- 6. Backward through qkv_proj = W_qkv @ x → d_x via W_qkv^T ----
        //
        // d_qkv_proj_all[t] now contains the complete gradient for token t because:
        //   - future-timestep conv contributions were written when t' > t was processed
        //   - current-timestep conv contribution was written in step 5 above
        //
        // g for in_proj_qkv = d_qkv_proj_all[t]  (pre-conv-SiLU linear-output gradient
        // summed over time, complete only after all future-t conv contributions land).
        let dx_t = &mut grad_inputs[t * hidden..(t + 1) * hidden];
        for j in 0..qkv_dim {
            let dq = d_qkv_proj_all[t * qkv_dim + j];
            if dq.abs() < f32::EPSILON {
                continue;
            }
            for i in 0..hidden {
                dx_t[i] += weights.in_proj_qkv[j * hidden + i] * dq;
            }
        }
        // LoRA weight grads for in_proj_qkv.
        // The LoRA forward added scale*(x@A_qkv^T)@B_qkv^T to qkv_proj[t].
        // That delta passed through conv1d+SiLU, whose backward already produced
        // d_qkv_proj_all[t] as the full gradient w.r.t. qkv_proj[t] (both base
        // and LoRA delta). So d_qkv_proj_all[t] is exactly the g for lora_vjp.
        // x = x_t (the original input), h = h_qkv_t.
        if have_lora {
            let d_qkv_g = &d_qkv_proj_all[t * qkv_dim..(t + 1) * qkv_dim];
            let h_qkv_t = &saved.h_qkv[t * lora_rank..(t + 1) * lora_rank];
            let (gb, ga, dx_lora) = lora_vjp(
                d_qkv_g,
                x_t,
                h_qkv_t,
                &saved.lora_a_qkv,
                &saved.lora_b_qkv,
                lora_rank,
                hidden,
                qkv_dim,
                lora_scale,
            );
            for k in 0..ga.len() {
                grad_a_qkv[k] += ga[k];
            }
            for k in 0..gb.len() {
                grad_b_qkv[k] += gb[k];
            }
            for j in 0..hidden {
                dx_t[j] += dx_lora[j];
            }
        }
    } // end reverse time loop

    GdnGrads {
        grad_a_qkv,
        grad_b_qkv,
        grad_a_z,
        grad_b_z,
        grad_a_b,
        grad_b_b,
        grad_a_a,
        grad_b_a,
        grad_a_out,
        grad_b_out,
        dx: grad_inputs,
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn l2_norm_sq(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn conv1d_silu_fwd(
    new_input: &[f32],
    conv_buffer: &mut [f32],
    conv_weight: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    let buf_len = kernel_size.saturating_sub(1);
    for ch in 0..conv_dim {
        let w_off = ch * kernel_size;
        let buf_off = ch * buf_len;
        let row = &mut conv_buffer[buf_off..buf_off + buf_len];

        let mut sum = 0.0f32;
        for t in 0..buf_len {
            sum += row[t] * conv_weight[w_off + t];
        }
        let x = new_input[ch];
        sum += x * conv_weight[w_off + buf_len];
        output[ch] = silu_f32(sum);

        if buf_len > 1 {
            row.copy_within(1..buf_len, 0);
        }
        if buf_len > 0 {
            row[buf_len - 1] = x;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::gdn::GatedDeltaNetWeights;
    use crate::model::qwen35_config::Qwen35Config;

    /// Simple xorshift RNG (deterministic, no deps).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 {
                0xDEAD_BEEF_CAFE_1234
            } else {
                seed
            })
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            let bits = (self.next() >> 40) as u32;
            let t = (bits as f32) / ((1u32 << 24) as f32);
            lo + t * (hi - lo)
        }
        fn fill(&mut self, out: &mut [f32], lo: f32, hi: f32) {
            for v in out {
                *v = self.f32_range(lo, hi);
            }
        }
    }

    /// Build minimal GDN weights for a small test fixture.
    fn make_tiny_weights(
        hidden: usize,
        num_kh: usize,
        num_vh: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        seed: u64,
    ) -> GatedDeltaNetWeights {
        let qkv_dim = num_kh * key_dim * 2 + num_vh * value_dim; // Q + K + V
        let output_dim = num_vh * value_dim;
        let mut rng = Rng::new(seed);
        let scale = 0.05;

        let mut in_proj_qkv = vec![0.0f32; qkv_dim * hidden];
        let mut in_proj_z = vec![0.0f32; output_dim * hidden];
        let mut in_proj_b = vec![0.0f32; num_kh * hidden];
        let mut in_proj_a = vec![0.0f32; num_kh * hidden];
        let mut conv1d_weight = vec![0.0f32; qkv_dim * kernel_size];
        let mut out_proj = vec![0.0f32; hidden * output_dim];
        let mut norm_weight = vec![0.0f32; value_dim];
        let mut a_log = vec![0.0f32; num_kh];
        let mut dt_bias = vec![0.0f32; num_kh];

        rng.fill(&mut in_proj_qkv, -scale, scale);
        rng.fill(&mut in_proj_z, -scale, scale);
        rng.fill(&mut in_proj_b, -scale, scale);
        rng.fill(&mut in_proj_a, -scale, scale);
        rng.fill(&mut conv1d_weight, -scale, scale);
        rng.fill(&mut out_proj, -scale, scale);
        for g in &mut norm_weight {
            *g = rng.f32_range(0.9, 1.1);
        }
        for a in &mut a_log {
            *a = rng.f32_range(-2.0, -0.5);
        }
        for dt in &mut dt_bias {
            *dt = rng.f32_range(-0.1, 0.1);
        }

        GatedDeltaNetWeights {
            in_proj_qkv,
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z,
            in_proj_z_rows: output_dim,
            in_proj_z_cols: hidden,
            in_proj_b,
            in_proj_b_rows: num_kh,
            in_proj_b_cols: hidden,
            in_proj_a,
            in_proj_a_rows: num_kh,
            in_proj_a_cols: hidden,
            a_log,
            dt_bias,
            conv1d_weight,
            conv_dim: qkv_dim,
            kernel_size,
            norm_weight,
            out_proj,
            out_proj_rows: hidden,
            out_proj_cols: output_dim,
        }
    }

    /// Build a Qwen35Config reflecting the tiny fixture dimensions.
    fn tiny_cfg(
        hidden: usize,
        num_kh: usize,
        num_vh: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
    ) -> Qwen35Config {
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.hidden_size = hidden;
        cfg.linear_num_key_heads = num_kh;
        cfg.linear_num_value_heads = Some(num_vh);
        cfg.linear_key_head_dim = key_dim;
        cfg.linear_value_head_dim = value_dim;
        cfg.linear_conv_kernel_dim = kernel_size;
        cfg
    }

    /// Compute loss = sum(w_i * y_i) where w_i are random fixed coefficients.
    /// This gives a constant grad_out (the w vector), eliminating near-zero gradients.
    fn linear_loss(outputs: &[f32], coeffs: &[f32]) -> f64 {
        outputs
            .iter()
            .zip(coeffs.iter())
            .map(|(&y, &w)| (y as f64) * (w as f64))
            .sum()
    }

    /// FD with a fixed linear loss (grad_out = coeffs).
    fn fd_grad_inputs_linear(
        inputs: &[f32],
        weights: &GatedDeltaNetWeights,
        cfg: &Qwen35Config,
        seq_len: usize,
        hidden: usize,
        eps: f64,
        coeffs: &[f32],
    ) -> Vec<f32> {
        let num_kh = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let kernel_size = cfg.linear_conv_kernel_dim;
        let scale = 1.0 / (key_dim as f32).sqrt();

        let mut grad = vec![0.0f32; inputs.len()];

        for idx in 0..inputs.len() {
            let mut inputs_p: Vec<f64> = inputs.iter().map(|&v| v as f64).collect();
            let mut inputs_m: Vec<f64> = inputs.iter().map(|&v| v as f64).collect();
            inputs_p[idx] += eps;
            inputs_m[idx] -= eps;

            // Forward in f32 (same as production) but cast to/from f64 at boundaries
            let inp_p_f32: Vec<f32> = inputs_p.iter().map(|&v| v as f32).collect();
            let inp_m_f32: Vec<f32> = inputs_m.iter().map(|&v| v as f32).collect();

            let mut saved_p = GdnSaved::new(
                seq_len,
                num_kh,
                value_heads,
                key_dim,
                value_dim,
                hidden,
                qkv_dim,
                output_dim,
                kernel_size,
                scale,
                cfg.rms_norm_eps,
            );
            let mut saved_m = GdnSaved::new(
                seq_len,
                num_kh,
                value_heads,
                key_dim,
                value_dim,
                hidden,
                qkv_dim,
                output_dim,
                kernel_size,
                scale,
                cfg.rms_norm_eps,
            );
            let mut out_p = vec![0.0f32; seq_len * hidden];
            let mut out_m = vec![0.0f32; seq_len * hidden];

            gdn_forward_save(
                &inp_p_f32,
                weights,
                cfg,
                &mut saved_p,
                &mut out_p,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0.0,
            );
            gdn_forward_save(
                &inp_m_f32,
                weights,
                cfg,
                &mut saved_m,
                &mut out_m,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0.0,
            );

            let lp = linear_loss(&out_p, coeffs);
            let lm = linear_loss(&out_m, coeffs);
            grad[idx] = ((lp - lm) / (2.0 * eps)) as f32;
        }
        grad
    }

    /// Run gradcheck with a linear loss.  Returns (max_rel_err, worst_idx, analytic[worst], fd[worst]).
    fn run_gradcheck_linear(
        hidden: usize,
        num_kh: usize,
        num_vh: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        seq_len: usize,
        input_scale: f32,
        weight_seed: u64,
        input_seed: u64,
        coeff_seed: u64,
        eps: f64,
    ) -> (f32, usize, f32, f32) {
        let cfg = tiny_cfg(hidden, num_kh, num_vh, key_dim, value_dim, kernel_size);
        let weights = make_tiny_weights(
            hidden,
            num_kh,
            num_vh,
            key_dim,
            value_dim,
            kernel_size,
            weight_seed,
        );
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let scale = 1.0 / (key_dim as f32).sqrt();

        let mut rng_in = Rng::new(input_seed);
        let mut inputs = vec![0.0f32; seq_len * hidden];
        rng_in.fill(&mut inputs, -input_scale, input_scale);

        // Random linear coefficients to guarantee non-trivial grads
        let mut rng_c = Rng::new(coeff_seed);
        let mut coeffs = vec![0.0f32; seq_len * hidden];
        rng_c.fill(&mut coeffs, -1.0, 1.0);

        // Analytic
        let mut saved = GdnSaved::new(
            seq_len,
            num_kh,
            num_vh,
            key_dim,
            value_dim,
            hidden,
            qkv_dim,
            output_dim,
            kernel_size,
            scale,
            cfg.rms_norm_eps,
        );
        let mut outputs = vec![0.0f32; seq_len * hidden];
        gdn_forward_save(
            &inputs,
            &weights,
            &cfg,
            &mut saved,
            &mut outputs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0.0,
        );
        let gdn_grads = gdn_backward(&coeffs, &saved, &weights);
        let analytic = gdn_grads.dx;

        // FD
        let fd = fd_grad_inputs_linear(&inputs, &weights, &cfg, seq_len, hidden, eps, &coeffs);

        // Compute max relative error (skip near-zero)
        let mut max_rel = 0.0f32;
        let mut worst = 0;
        let mut n_tested = 0usize;
        for i in 0..analytic.len() {
            let abs_err = (analytic[i] - fd[i]).abs();
            // Skip indices where both are negligibly small.  With f32 forward
            // arithmetic and eps=1e-3, the FD oracle itself has ~1% relative
            // error on gradients of magnitude ≤1e-4.  Only test where the
            // signal is large enough that FD is accurate.
            let mag = fd[i].abs().max(analytic[i].abs());
            if mag < 1e-4 {
                continue;
            }
            n_tested += 1;
            let rel = abs_err / mag;
            if rel > max_rel {
                max_rel = rel;
                worst = i;
            }
        }
        assert!(
            n_tested >= 10,
            "gradcheck skipped too many entries (only {n_tested} with |g| >= 1e-4); \
             increase input_scale or weight_scale to produce meaningful gradients"
        );
        (max_rel, worst, analytic[worst], fd[worst])
    }

    #[test]
    fn gradcheck_gdn_backward() {
        // Tiny fixture: seq=8, hidden=32, 1 key-head, 1 value-head, key_dim=8, value_dim=8
        // Linear loss with random coefficients → every output component contributes
        // to the gradient, avoiding near-zero masking issues.
        //
        // eps=1e-3: large enough to avoid f32 catastrophic cancellation in the
        // FD oracle, while still giving an accurate finite-difference estimate
        // for gradients with magnitude >1e-4.  For near-zero gradients (skipped
        // by the mag < 1e-5 guard) FD noise is irrelevant.
        let (max_rel, worst, analytic_v, fd_v) = run_gradcheck_linear(
            32, 1, 1, 8, 8, 3, 8, 0.5,  // input_scale
            42,   // weight_seed
            1337, // input_seed
            999,  // coeff_seed
            1e-3, // eps
        );

        assert!(
            max_rel < 1e-2,
            "gradcheck FAILED: max rel-err = {max_rel:.2e} at index {worst} \
             (analytic={analytic_v}, fd={fd_v})",
        );
    }

    #[test]
    fn gradcheck_gdn_backward_multi_head() {
        // 2 key-heads, 4 value-heads (ratio=2), seq=4
        let (max_rel, worst, analytic_v, fd_v) = run_gradcheck_linear(
            32, 2, 4, 6, 6, 2, 4, 0.5,  // input_scale
            99,   // weight_seed
            2024, // input_seed
            777,  // coeff_seed
            1e-3, // eps
        );

        assert!(
            max_rel < 1e-2,
            "multi-head gradcheck FAILED: max rel-err = {max_rel:.2e} at index {worst} \
             (analytic={analytic_v}, fd={fd_v})",
        );
    }

    #[test]
    fn zero_grad_output_yields_zero_input_grad() {
        let hidden = 16;
        let num_kh = 1;
        let num_vh = 1;
        let key_dim = 4;
        let value_dim = 4;
        let kernel_size = 2;
        let seq_len = 3;

        let cfg = tiny_cfg(hidden, num_kh, num_vh, key_dim, value_dim, kernel_size);
        let weights = make_tiny_weights(hidden, num_kh, num_vh, key_dim, value_dim, kernel_size, 7);
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let scale = 1.0 / (key_dim as f32).sqrt();

        let mut rng = Rng::new(888);
        let mut inputs = vec![0.0f32; seq_len * hidden];
        rng.fill(&mut inputs, -0.5, 0.5);

        let mut saved = GdnSaved::new(
            seq_len,
            num_kh,
            num_vh,
            key_dim,
            value_dim,
            hidden,
            qkv_dim,
            output_dim,
            kernel_size,
            scale,
            cfg.rms_norm_eps,
        );
        let mut outputs = vec![0.0f32; seq_len * hidden];
        gdn_forward_save(
            &inputs,
            &weights,
            &cfg,
            &mut saved,
            &mut outputs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0.0,
        );

        let grad_out = vec![0.0f32; seq_len * hidden];
        let gdn_grads = gdn_backward(&grad_out, &saved, &weights);
        let analytic = gdn_grads.dx;

        for (i, &v) in analytic.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "zero upstream grad should give zero input grad at [{i}], got {v}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // LoRA weight-grad gradcheck
    // ---------------------------------------------------------------------------
    //
    // Finite-difference check for the 10 LoRA weight gradients (grad_a_*/grad_b_*).
    // The existing gradcheck_gdn_backward tests only cover dx with LoRA OFF.
    // This test runs gdn_forward_save with LoRA ON and checks that the weight grads
    // returned by gdn_backward agree with central-difference estimates.
    //
    // DO NOT RUN this test manually until the AM gradcheck pass is scheduled.
    // (Compile-gate only — verifies the test compiles under --all-targets.)

    /// Helper: run one forward + backward with the given LoRA arrays and return the
    /// full GdnGrads.  All ten LoRA slices must be Some.
    #[allow(clippy::too_many_arguments)]
    fn run_lora_forward_backward(
        inputs: &[f32],
        weights: &GatedDeltaNetWeights,
        cfg: &Qwen35Config,
        seq_len: usize,
        coeffs: &[f32],
        lora_rank: usize,
        lora_scale: f32,
        a_qkv: &[f32],
        b_qkv: &[f32],
        a_z: &[f32],
        b_z: &[f32],
        a_b: &[f32],
        b_b: &[f32],
        a_a: &[f32],
        b_a: &[f32],
        a_out: &[f32],
        b_out: &[f32],
    ) -> GdnGrads {
        let hidden = cfg.hidden_size;
        let num_kh = cfg.linear_num_key_heads;
        let num_vh = cfg.linear_num_value_heads();
        let key_dim = cfg.linear_key_head_dim;
        let value_dim = cfg.linear_value_head_dim;
        let kernel_size = cfg.linear_conv_kernel_dim;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let scale = 1.0 / (key_dim as f32).sqrt();

        let mut saved = GdnSaved::new(
            seq_len,
            num_kh,
            num_vh,
            key_dim,
            value_dim,
            hidden,
            qkv_dim,
            output_dim,
            kernel_size,
            scale,
            cfg.rms_norm_eps,
        );
        let mut outputs = vec![0.0f32; seq_len * hidden];
        gdn_forward_save(
            inputs,
            weights,
            cfg,
            &mut saved,
            &mut outputs,
            Some(a_qkv),
            Some(b_qkv),
            Some(a_z),
            Some(b_z),
            Some(a_b),
            Some(b_b),
            Some(a_a),
            Some(b_a),
            Some(a_out),
            Some(b_out),
            lora_rank,
            lora_scale,
        );
        gdn_backward(coeffs, &saved, weights)
    }

    /// Central-difference estimate of d(loss)/d(param[idx]) for a single entry.
    #[allow(clippy::too_many_arguments)]
    fn fd_lora_weight_entry(
        inputs: &[f32],
        weights: &GatedDeltaNetWeights,
        cfg: &Qwen35Config,
        seq_len: usize,
        coeffs: &[f32],
        lora_rank: usize,
        lora_scale: f32,
        a_qkv: &[f32],
        b_qkv: &[f32],
        a_z: &[f32],
        b_z: &[f32],
        a_b: &[f32],
        b_b: &[f32],
        a_a: &[f32],
        b_a: &[f32],
        a_out: &[f32],
        b_out: &[f32],
        // Which of the 10 arrays to perturb, and which entry:
        which: usize, // 0=a_qkv 1=b_qkv 2=a_z 3=b_z 4=a_b 5=b_b 6=a_a 7=b_a 8=a_out 9=b_out
        idx: usize,
        eps: f32,
    ) -> f32 {
        // Helper: clone an array, perturb entry [idx] by delta, re-run, return loss.
        let perturbed_loss = |delta: f32| -> f32 {
            let mut aq = a_qkv.to_vec();
            let mut bq = b_qkv.to_vec();
            let mut az = a_z.to_vec();
            let mut bz = b_z.to_vec();
            let mut ab = a_b.to_vec();
            let mut bb = b_b.to_vec();
            let mut aa = a_a.to_vec();
            let mut ba = b_a.to_vec();
            let mut ao = a_out.to_vec();
            let mut bo = b_out.to_vec();
            let arr: &mut Vec<f32> = match which {
                0 => &mut aq,
                1 => &mut bq,
                2 => &mut az,
                3 => &mut bz,
                4 => &mut ab,
                5 => &mut bb,
                6 => &mut aa,
                7 => &mut ba,
                8 => &mut ao,
                _ => &mut bo,
            };
            arr[idx] += delta;
            let hidden = cfg.hidden_size;
            let num_kh = cfg.linear_num_key_heads;
            let num_vh = cfg.linear_num_value_heads();
            let key_dim = cfg.linear_key_head_dim;
            let value_dim = cfg.linear_value_head_dim;
            let kernel_size = cfg.linear_conv_kernel_dim;
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let scale = 1.0 / (key_dim as f32).sqrt();
            let mut saved = GdnSaved::new(
                seq_len,
                num_kh,
                num_vh,
                key_dim,
                value_dim,
                hidden,
                qkv_dim,
                output_dim,
                kernel_size,
                scale,
                cfg.rms_norm_eps,
            );
            let mut outputs = vec![0.0f32; seq_len * hidden];
            gdn_forward_save(
                inputs,
                weights,
                cfg,
                &mut saved,
                &mut outputs,
                Some(&aq),
                Some(&bq),
                Some(&az),
                Some(&bz),
                Some(&ab),
                Some(&bb),
                Some(&aa),
                Some(&ba),
                Some(&ao),
                Some(&bo),
                lora_rank,
                lora_scale,
            );
            outputs
                .iter()
                .zip(coeffs.iter())
                .map(|(&y, &c)| y * c)
                .sum()
        };
        let lp = perturbed_loss(eps);
        let lm = perturbed_loss(-eps);
        (lp - lm) / (2.0 * eps)
    }

    // Core helper that exercises weight-grad gradcheck for any fixture.
    // Returns (max_rel_err, n_tested).
    #[allow(clippy::too_many_arguments)]
    fn run_lora_weight_gradcheck(
        hidden: usize,
        num_kh: usize,
        num_vh: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        seq_len: usize,
        lora_rank: usize,
        lora_scale: f32,
        weight_seed: u64,
        input_seed: u64,
        lora_seed: u64,
        coeff_seed: u64,
        eps: f32,
    ) -> (f32, usize) {
        let cfg = tiny_cfg(hidden, num_kh, num_vh, key_dim, value_dim, kernel_size);
        let weights = make_tiny_weights(
            hidden,
            num_kh,
            num_vh,
            key_dim,
            value_dim,
            kernel_size,
            weight_seed,
        );
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();

        // Inputs
        let mut rng_in = Rng::new(input_seed);
        let mut inputs = vec![0.0f32; seq_len * hidden];
        rng_in.fill(&mut inputs, -0.5, 0.5);

        // Upstream coeffs (linear loss)
        let mut rng_c = Rng::new(coeff_seed);
        let mut coeffs = vec![0.0f32; seq_len * hidden];
        rng_c.fill(&mut coeffs, -1.0, 1.0);

        // LoRA matrices — small random values so grad_A and grad_B are non-vacuous.
        // Shapes: A=[rank, d_in], B=[d_out, rank] for each projection.
        let mut rng_l = Rng::new(lora_seed);
        let mut a_qkv = vec![0.0f32; lora_rank * hidden]; // [rank, hidden]
        let mut b_qkv = vec![0.0f32; qkv_dim * lora_rank]; // [qkv_dim, rank]
        let mut a_z = vec![0.0f32; lora_rank * hidden]; // [rank, hidden]
        let mut b_z = vec![0.0f32; output_dim * lora_rank]; // [output_dim, rank]
        let mut a_b = vec![0.0f32; lora_rank * hidden]; // [rank, hidden]
        let mut b_b = vec![0.0f32; num_kh * lora_rank]; // [num_kh, rank]
        let mut a_a = vec![0.0f32; lora_rank * hidden]; // [rank, hidden]
        let mut b_a = vec![0.0f32; num_kh * lora_rank]; // [num_kh, rank]
        let mut a_out = vec![0.0f32; lora_rank * output_dim]; // [rank, output_dim]
        let mut b_out = vec![0.0f32; hidden * lora_rank]; // [hidden, rank]
        for arr in [
            &mut a_qkv, &mut b_qkv, &mut a_z, &mut b_z, &mut a_b, &mut b_b, &mut a_a, &mut b_a,
            &mut a_out, &mut b_out,
        ]
        .iter_mut()
        {
            rng_l.fill(arr, -0.05, 0.05);
        }

        // Analytic weight grads via forward+backward.
        let grads = run_lora_forward_backward(
            &inputs, &weights, &cfg, seq_len, &coeffs, lora_rank, lora_scale, &a_qkv, &b_qkv, &a_z,
            &b_z, &a_b, &b_b, &a_a, &b_a, &a_out, &b_out,
        );

        // Collect (which_array, analytic_grad_slice, array_len) triples.
        let analytic_arrays: [(&[f32], usize); 10] = [
            (&grads.grad_a_qkv, a_qkv.len()),
            (&grads.grad_b_qkv, b_qkv.len()),
            (&grads.grad_a_z, a_z.len()),
            (&grads.grad_b_z, b_z.len()),
            (&grads.grad_a_b, a_b.len()),
            (&grads.grad_b_b, b_b.len()),
            (&grads.grad_a_a, a_a.len()),
            (&grads.grad_b_a, b_a.len()),
            (&grads.grad_a_out, a_out.len()),
            (&grads.grad_b_out, b_out.len()),
        ];

        let mut max_rel = 0.0f32;
        let mut n_tested = 0usize;

        for (which, (analytic_slice, arr_len)) in analytic_arrays.iter().enumerate() {
            // Sample up to 6 entries spread across each array.
            let step = (arr_len / 6).max(1);
            let mut idx = 0usize;
            while idx < *arr_len {
                let analytic_v = analytic_slice[idx];
                let fd_v = fd_lora_weight_entry(
                    &inputs, &weights, &cfg, seq_len, &coeffs, lora_rank, lora_scale, &a_qkv,
                    &b_qkv, &a_z, &b_z, &a_b, &b_b, &a_a, &b_a, &a_out, &b_out, which, idx, eps,
                );
                let mag = fd_v.abs().max(analytic_v.abs());
                if mag >= 1e-4 {
                    n_tested += 1;
                    let rel = (fd_v - analytic_v).abs() / mag;
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
                idx += step;
            }
        }

        (max_rel, n_tested)
    }

    #[test]
    fn gradcheck_gdn_lora_weight_grads() {
        // Tiny GDN: hidden=32, num_kh=1, value_heads=1, key_dim=8, value_dim=8,
        // kernel_size=3, seq=4; rank=2; scale=0.5.
        let (max_rel, n_tested) = run_lora_weight_gradcheck(
            32, 1, 1, 8, 8, 3, 4,    // hidden, num_kh, num_vh, key_dim, value_dim, kernel, seq
            2,    // lora_rank
            0.5,  // lora_scale
            42,   // weight_seed
            1337, // input_seed
            777,  // lora_seed
            999,  // coeff_seed
            1e-3, // eps
        );
        assert!(
            n_tested >= 10,
            "lora weight gradcheck: too few testable entries ({n_tested}); \
             increase array sizes or tighten threshold"
        );
        assert!(
            max_rel < 1e-2,
            "gdn lora weight gradcheck failed: max_rel={max_rel:.3e} \
             (n_tested={n_tested})"
        );
    }

    #[test]
    fn gradcheck_gdn_lora_weight_grads_multi_head() {
        // Multi-head variant: num_kh=2, value_heads=4 exercises alpha/beta head-sharing reduction.
        let (max_rel, n_tested) = run_lora_weight_gradcheck(
            32, 2, 4, 8, 8, 3,
            4,    // hidden, num_kh=2, num_vh=4, key_dim=8, value_dim=8, kernel, seq
            2,    // lora_rank
            0.5,  // lora_scale
            99,   // weight_seed
            2024, // input_seed
            555,  // lora_seed
            111,  // coeff_seed
            1e-3, // eps
        );
        assert!(
            n_tested >= 10,
            "lora weight gradcheck (multi-head): too few testable entries ({n_tested})"
        );
        assert!(
            max_rel < 1e-2,
            "gdn lora weight gradcheck (multi-head) failed: max_rel={max_rel:.3e} \
             (n_tested={n_tested})"
        );
    }
}
