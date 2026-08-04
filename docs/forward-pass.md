# Qwen3.5 Forward-Pass Walkthrough

This document traces a token through the Qwen3.5 inference forward pass,
referencing the real module and function names from `crates/inference/src/`.
The path described here is the **Qwen3.5 path** (`crate::model::qwen35`).

---

## Step-by-step trace

### 1. Generation entry — `crate::model::qwen35::Qwen35Model::generate`

The caller invokes `Qwen35Model::generate` with a prompt string and a
`GenerateConfig`. The method tokenizes the prompt and copies the resulting
token ids into a local buffer.

### 2. RNG initialization — `crate::model::qwen35::generation::initial_rng_state`

`initial_rng_state` initializes a seeded or clock-derived RNG state. This
state is threaded through all subsequent sampling calls.

### 3. State allocation — `crate::model::qwen35::Qwen35Model::generate`

`generate` allocates three per-session data structures:

- `GatedDeltaNetState` — recurrent state for GDN (linear-attention) layers.
- `KvCache` — key/value cache for full-attention layers.
- `ForwardScratch` — reusable activation buffers (hidden, q, k, v, …).

### 4. Prompt prefill — `Qwen35Model::prefill_tokens_batched_for_generate`

For dense models, `generate` first calls `prefill_tokens_batched_for_generate`,
which processes all prompt positions in one batched layer pass and returns only
the final position's logits. MoE models return `UnsupportedModel` from that path
before mutating recurrent state, so generation falls back to the serial
`prefill_tokens` loop. The serial loop calls `forward_step(token_id, pos, …)`
once per prompt token while populating the KV cache and GDN state.

### 5. Single-token forward entry — `crate::model::qwen35::Qwen35Model::forward_step`

`forward_step` is the core per-token entry point used for decode and the serial
MoE prefill fallback. It grows scratch buffer capacity if the current token
position requires it.

### 6. Embedding lookup — `crate::model::qwen35::Qwen35Model::forward_step`

The embedding lookup is inline: the row for `token_id` is copied from
`self.weights.embed_tokens` directly into `scratch.hidden`.

### 7. Layer loop entry — `crate::model::qwen35::Qwen35Model::forward_step`

`forward_step` loops over all hidden layers. For each layer it saves the
current `scratch.hidden` as the residual stream and applies the
pre-attention RMSNorm.

### 8. Pre-attention norm — `crate::model::qwen35::norm::qwen35_rms_norm`

`qwen35_rms_norm` is a shifted RMSNorm: it computes
`x * inv_rms * (1.0 + gamma)` per element, where `x` is the input activation
and `gamma` is the learned weight.

### 9. Attention dispatch — `crate::model::qwen35::Qwen35Model::run_attention_layer`

`run_attention_layer` dispatches on the layer's attention type:

- `AttentionWeights::Linear` → GDN branch (step 10).
- `AttentionWeights::Full` → full-attention branch (step 11).

### 10. GDN branch — `crate::attention::gdn_fused::gated_delta_net_step_fused`

For linear-attention layers, `gated_delta_net_step_fused`:

1. Projects Q, K, V, Z, beta, and alpha from the normed hidden state.
2. Runs fused conv1d + SiLU on the input features.
3. Updates the recurrent `GatedDeltaNetState`.
4. Output-projects the result back to the residual dimension.

### 11. Full-attention branch — `crate::model::qwen35::Qwen35Model::full_attention_step_from_attn_out`

For full-attention layers:

1. Projects Q, K, V (step 12).
2. Applies Rotary Position Embeddings (RoPE) to Q and K.
3. Appends the new K/V to the `KvCache`.
4. Computes the attention context (step 13).
5. Gates and output-projects the context.

### 12. Q/K/V projection — `crate::model::qwen35::Qwen35Model::project_qkv`

`project_qkv` runs three matmuls. Because `attn_output_gate` is enabled, the
Q projection weight is `[2*q_dim, hidden]`: it produces both `Q` and the output
gate `Z` in a single matmul, which `split_q_and_gate` deinterleaves. The other
two matmuls produce K and V. Per-head Q/K RMSNorm is applied to each resulting
head vector; `gate_Z` is applied later as the sigmoid output gate.

### 13. Attention context — `crate::model::qwen35::Qwen35Model::compute_attention_context`

`compute_attention_context` computes scaled dot-product attention scores over
the cached K vectors, applies causal softmax, and accumulates the weighted V
vectors into `scratch.context`.

### 14. Attention residual and post-attention norm — `crate::model::qwen35::Qwen35Model::forward_step`

Back in `forward_step`, the attention output is added back to the residual
stream and `qwen35_rms_norm` is applied again (post-attention norm).

### 15. FFN dispatch — `crate::model::qwen35::Qwen35Model::run_ffn_layer`

`run_ffn_layer` dispatches on the layer's FFN type:

- Dense FFN → `dense_ffn_step_from_ffn_out` (step 16).
- MoE FFN → `moe_ffn_step` (step 17).

### 16. Dense MLP — `crate::model::qwen35::Qwen35Model::dense_ffn_step_from_ffn_out`

The dense path:

1. Runs gate and up matmuls on the normed hidden state.
2. Applies SiLU to the gate output and elementwise-multiplies with the up output.
3. Down-projects the result to the residual dimension.

### 17. MoE FFN — `crate::model::qwen35::moe::moe_ffn_step`

The Mixture-of-Experts path:

1. Computes router probabilities from the normed hidden state.
2. Selects and renormalizes the top-k expert scores.
3. Accumulates each selected routed expert's contribution.
4. Adds the shared expert's output.

### 18. MLP residual — `crate::model::qwen35::Qwen35Model::forward_step`

The FFN output is added back to the residual stream.

### 19. Final RMSNorm — `crate::model::qwen35::Qwen35Model::forward_step`

After the last layer, `forward_step` applies a final `qwen35_rms_norm` using
`self.weights.final_norm`.

### 20. LM head weight selection — `crate::model::qwen35::weights::ModelWeights::logits_weight`

`logits_weight` selects the vocabulary projection weights: the explicit
`lm_head` weight tensor if present, otherwise the tied `embed_tokens` tensor.

### 21. Logit computation — `crate::forward::cpu::matmul_bt`

`matmul_bt` (transposed-B matmul) multiplies the final hidden state against
the vocabulary weight matrix, producing a logit vector of size `vocab_size`.

### 22–23. Sampling — `crate::model::qwen35::sampling::sample_token`

`sample_token` is the Qwen3.5 sampling entry point, called once after the
completed prompt prefill and again on each decode step.

It applies, in order:

1. Repetition penalty on the logit of recently generated tokens.
2. Greedy fallback when temperature is zero or degenerate.
3. Temperature scaling of the logits (skipped when temperature is 1.0).
4. Top-k filtering (retains the k highest logits).
5. Softmax followed by top-p (nucleus) filtering.
6. Draws a token from the resulting distribution.

The drawn token id is appended to the generated sequence and becomes the
input to the next `forward_step` call.
