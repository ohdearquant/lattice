use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
use crate::model::qwen35_config::Qwen35Config;

/// Simple KV cache for autoregressive decoding (full-attention layers only).
pub(crate) struct KvCache {
    /// k_cache[layer_idx] is a Vec storing all K vectors: flat [seq_len, kv_dim]
    pub(crate) k: Vec<Vec<f32>>,
    /// v_cache[layer_idx] is a Vec storing all V vectors: flat [seq_len, kv_dim]
    pub(crate) v: Vec<Vec<f32>>,
    /// Current sequence length in cache.
    pub(crate) seq_len: usize,
}

impl KvCache {
    pub(crate) fn new(num_full_layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); num_full_layers],
            v: vec![Vec::new(); num_full_layers],
            seq_len: 0,
        }
    }

    pub(crate) fn append_kv(&mut self, full_layer_idx: usize, k_vec: &[f32], v_vec: &[f32]) {
        self.k[full_layer_idx].extend_from_slice(k_vec);
        self.v[full_layer_idx].extend_from_slice(v_vec);
    }

    pub(crate) fn reserve(&mut self, max_seq_len: usize, kv_dim: usize) {
        let needed = max_seq_len * kv_dim;
        for k in &mut self.k {
            k.reserve_exact(needed.saturating_sub(k.capacity()));
        }
        for v in &mut self.v {
            v.reserve_exact(needed.saturating_sub(v.capacity()));
        }
    }
}

pub(crate) struct ForwardScratch {
    pub(crate) hidden: Vec<f32>,
    pub(crate) residual: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    // Full-attention buffers
    pub(crate) q_buf: Vec<f32>,
    pub(crate) k_buf: Vec<f32>,
    pub(crate) v_buf: Vec<f32>,
    pub(crate) scores: Vec<f32>,
    pub(crate) context: Vec<f32>,
    // MLP buffers
    pub(crate) gate_buf: Vec<f32>,
    pub(crate) up_buf: Vec<f32>,
    pub(crate) ffn_out: Vec<f32>,
    pub(crate) input_tmp: Vec<f32>,
    pub(crate) q_and_gate: Vec<f32>,
    pub(crate) gate_z: Vec<f32>,
    pub(crate) down_input: Vec<f32>,
    // GatedDeltaNet scratch
    pub(crate) gdn_scratch: GatedDeltaNetFusedScratch,
    // Logits buffer
    pub(crate) logits: Vec<f32>,
    // Q8_0 activation quantization scratch — reused across matmul_q8_neon_into calls
    pub(crate) x_q_scratch: Vec<i8>,
    // MoE scratch buffers
    pub(crate) router_logits: Vec<f32>,            // [num_experts]
    pub(crate) router_selected: Vec<(usize, f32)>, // [num_experts_per_tok]
    pub(crate) expert_out: Vec<f32>,               // [hidden]
}

impl ForwardScratch {
    pub(crate) fn new() -> Self {
        Self {
            hidden: Vec::new(),
            residual: Vec::new(),
            attn_out: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            scores: Vec::new(),
            context: Vec::new(),
            gate_buf: Vec::new(),
            up_buf: Vec::new(),
            ffn_out: Vec::new(),
            input_tmp: Vec::new(),
            q_and_gate: Vec::new(),
            gate_z: Vec::new(),
            down_input: Vec::new(),
            gdn_scratch: GatedDeltaNetFusedScratch::default(),
            logits: Vec::new(),
            x_q_scratch: Vec::new(),
            router_logits: Vec::new(),
            router_selected: Vec::new(),
            expert_out: Vec::new(),
        }
    }

    pub(crate) fn ensure_capacity(&mut self, cfg: &Qwen35Config, max_kv_len: usize) {
        let h = cfg.hidden_size;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size();
        let shared_inter = cfg.shared_expert_intermediate_size();
        let inter = dense_inter.max(moe_inter).max(shared_inter);
        let max_q8_input = h.max(inter).max(q_dim).max(cfg.linear_output_dim());

        resize(&mut self.hidden, h);
        resize(&mut self.residual, h);
        resize(&mut self.attn_out, h);
        resize(&mut self.q_buf, q_dim);
        resize(&mut self.k_buf, kv_dim);
        resize(&mut self.v_buf, kv_dim);
        resize(&mut self.scores, cfg.num_attention_heads * (max_kv_len + 1));
        resize(&mut self.context, q_dim);
        resize(&mut self.gate_buf, inter);
        resize(&mut self.up_buf, inter);
        resize(&mut self.ffn_out, h);
        resize(&mut self.expert_out, h);
        resize(&mut self.router_logits, cfg.num_experts.unwrap_or(0));
        if self.router_selected.len() < cfg.num_experts_per_tok.unwrap_or(0) {
            self.router_selected
                .resize(cfg.num_experts_per_tok.unwrap_or(0), (usize::MAX, 0.0));
        }
        self.ensure_decode_capacity(h, 2 * q_dim, q_dim, inter);
        if self.x_q_scratch.len() < max_q8_input {
            self.x_q_scratch.resize(max_q8_input, 0);
        }
        resize(&mut self.logits, cfg.vocab_size);
    }

    /// Deinterleave packed `[Q|gate]` from `q_and_gate` into `q_buf` and `gate_z`.
    ///
    /// Zero-allocation — destructures `self` into disjoint field borrows and
    /// delegates to [`deinterleave_q_gate`].
    ///
    /// # Panics
    ///
    /// Panics if any buffer is too short for `num_heads * head_dim`.
    pub(crate) fn split_q_and_gate(&mut self, num_heads: usize, head_dim: usize) {
        let q_dim = num_heads * head_dim;
        let Self {
            q_and_gate,
            q_buf,
            gate_z,
            ..
        } = self;
        crate::attention::gated::deinterleave_q_gate(
            &q_and_gate[..2 * q_dim],
            &mut q_buf[..q_dim],
            &mut gate_z[..q_dim],
            num_heads,
            head_dim,
        );
    }

    pub(crate) fn ensure_decode_capacity(
        &mut self,
        hidden: usize,
        q_proj_dim: usize,
        q_dim: usize,
        inter: usize,
    ) {
        resize(&mut self.input_tmp, hidden);
        resize(&mut self.q_and_gate, q_proj_dim);
        resize(&mut self.gate_z, q_dim);
        resize(&mut self.down_input, inter);
    }
}

#[inline]
pub(crate) fn resize(buf: &mut Vec<f32>, n: usize) {
    if buf.len() < n {
        buf.resize(n, 0.0);
    }
}
