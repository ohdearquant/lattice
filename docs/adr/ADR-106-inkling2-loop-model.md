# ADR-106: Running Inkling 2 Mini, a Looped Mixture-of-Experts Text Model, on the CPU

**Status**: Proposed
**Date**: 2026-11-23
**Crates**: lattice-model-inkling2 (new), lattice-arch, lattice-weights, lattice-tokenizer, lattice-serve
**Depends on**: ADR-092, ADR-100, ADR-103

## Context

Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, Apache-2.0) is a mixture-of-experts model of about 26 billion parameters with a text decoder and a vision tower. Its text decoder differs from every decoder lattice runs today:

- Layers 6 to 17 run three times per token, and the passes after the first read messages from the previous pass.
- Each attention and MLP input is an attention-weighted mixture of earlier block states and the current state, not a residual sum.
- There is no rotary embedding. Position enters through a content-conditioned relative bias, L2 clips on queries and keys, a logarithmic query scale in the global layers, per-head attention sinks and a gated output projection.
- Keys, values, the attention output and the MLP output pass through short causal convolutions; in the global layers the key and value convolutions have per-token taps.
- The 128 experts per layer are stored as NVFP4 in layers 0 to 19 and MXFP8 in layers 20 to 23.

The only executable description of the model is the Hugging Face transformers implementation. This ADR records what that implementation does, by file and line, so that lattice's implementation, written in lattice's own terms, can be checked against it by a reader who runs nothing.

### Reference files

All at transformers commit `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef`. Line numbers refer to these files at that commit.

| Tag | File                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------- |
| [M] | `modeling_inkling2_loop.py` (the Inkling 2 loop model, under `src/transformers/models/`)                |
| [C] | `configuration_inkling2_loop.py` (its configuration class, same directory)                              |
| [L] | `src/transformers/core_model_loading.py` (expert dequantization and key conversions on load)            |
| [K] | `src/transformers/conversion_mapping.py` (checkpoint key renames)                                       |
| [S] | the JSON header of the checkpoint's `model-00001-of-00001.safetensors` (names, dtypes, shapes; no data) |
| [W] | `src/transformers/masking_utils.py` (sliding-window and causal masks)                                   |
| [Q] | `src/transformers/cache_utils.py` (key, value and convolution caches)                                   |

## Decision

1. A new crate, `lattice-model-inkling2` at `crates/models/inkling2`, owns the Inkling 2 text decoder, as the other model crates own theirs under ADR-103. It depends on `lattice-arch` for the configuration, `lattice-weights` for safetensors access and the packed NVFP4 and MXFP8 matrix-vector products, and `lattice-tokenizer` for the tokenizer. Serving goes through the model-owned decoder boundary of ADR-092.
2. **Delivery bar.** The model is delivered when lattice generates text from the published checkpoint through `lattice-serve` on the CPU, and the first generated token matches the reference's greedy first token on a fixed prompt set. Metal execution, multi-token prediction, the vision tower and image inputs are outside the bar. CPU speed is reported and not gated.
3. Experts stay packed in memory and are multiplied by the packed matrix-vector kernels. Dequantizing to dense weights on load, as the reference does, is ruled out by memory (see Memory).
4. Non-expert tensors stay in their stored dtype (bf16 for the projections, embeddings and norms; f32 for the mixture queries and the router bias) and are widened to f32 where they are read. Arithmetic is f32 throughout.
5. The work lands in five steps, each its own pull request: the configuration parser (`lattice-arch`), the packed matrix-vector products (`lattice-weights`), the tokenizer's pre-tokenizer pattern (`lattice-tokenizer`, issue #2911), the CPU forward in the new crate, and the serve provider.

## Reference behaviour

Notation: `n(x)` is RMS normalization without a weight, `x * rsqrt(mean(x^2) + eps)`, computed in f32 ([M] 67-76). `RMSNorm_w(x)` is the same multiplied by a learned weight `w`, with no `1 + w` offset ([M] 730-748). `softcap_c(x) = c * tanh(x / c)` ([M] 751-757). H = 2304 is the hidden size.

### Layer schedule

- 24 layers. Layer `i` is global when `(i + 1) % 6 == 0` (layers 5, 11, 17, 23) and sliding otherwise ([C] 145-153).
- Layers 6 to 17 form the loop core and run `num_passes = 3` times ([C] 145-153), so one token takes 6 + 12 x 3 + 6 = 48 layer applications ([C] 172-176).
- Block boundaries fall on layers where `i % 6 == 0` (0, 6, 12, 18) ([M] 861).

### The image token default

The published `config.json` carries no `image_token_id`. The parser takes 200054, the default of the reference's top-level configuration class ([C] 183, 200). The checkpoint's `processor_config.json` names the image token `<|unused_200054|>`, and its `tokenizer_config.json` maps id 200054 to that string. Geometry is never defaulted; this id is, because the published file omits it. The text path does not read it.

### a) The loop, its messages and its caches

**Forward** ([M] 1308-1447):

1. `x = RMSNorm_embed_norm(embed[token])` ([M] 1240-1246, 1322). `block_states` starts empty ([M] 1359-1360).
2. Layers 0 to 5 run once. The input of each boundary layer is appended to `block_states` ([M] 1362-1376). The hidden state and block states after layer 5 are kept as the pre-loop state ([M] 1378-1379).
3. Pass 1 runs layers 6 to 17 with no messages ([M] 1381-1395).
4. Passes 2 and 3 each restart from the pre-loop hidden state and pre-loop block states ([M] 1402). Passes do not chain through the residual stream; they communicate only through messages.
5. Layers 18 to 23 run once on the output and block states of the last pass ([M] 1426-1440).
6. The output is `RMSNorm_norm(A_out(block_states, x))`, where `A_out` is the final block-attention mixture ([M] 1442-1443), then the unembedding and the logit softcap (see f).

**Messages.** In passes 2 and 3, the message source is two states from the previous pass: the block state it appended at layer 12 (the one at layer 6 is skipped, being the pre-loop state) and its final hidden state, the output of layer 17 ([M] 1400-1401). For loop layer `j` (0 to 11), four learned queries `Q_j` of shape [4, H] (query, key, value, MLP) attend over the RMS-normalized source states, with the same normalized states as values, scale `H^-0.5`, and a softmax over the two states ([M] 145-173, 1407). Each message is therefore a convex combination of normalized states.

**Gates.** Each loop layer has four message gates ([M] 931-934): query, key, value and MLP. A gate computes `z = D [n(h); n(m)]` with `D` of shape [64, 2H], hidden state first, then `out = (1 - tanh(F z)) * h + tanh(G z) * m` with `F` and `G` of shape [H, 64], in f32 ([M] 79-96). The un-normalized `h` and `m` enter the output. `loop_gate_rank = 64` is the width of `z`.

**Where the gates apply.** In a sliding loop layer, the attention input mixture is gated three times, once per message, before the input norm: query, key and value then take separate inputs ([M] 959-967). In a global loop layer, each of the three attention-input mixtures is gated by its message before its own input norm ([M] 1113-1119). In both, the MLP input mixture is gated by the MLP message before the post-attention norm ([M] 990-993, 1142-1145). In pass 1 there are no messages and no gating.

**Caches.** Each pass of the loop core has its own key/value cache and its own convolution states: the reference appends two copies of the loop layers' cache entries and offsets each pass into them ([M] 1354-1357, 1405). One decode step therefore touches 24 + 12 x 2 = 48 cache slots, each holding keys, values and four convolution states (conv indices 0 key, 1 value, 2 post-attention, 3 post-MLP: [M] 562-567, 868-873).

**Checkpoint names** ([K] 2684-2708): `controllers.{attn,k,v,mlp}` are the query, key, value and MLP gates (`attn` maps to the query gate); `down_weight` is `D`, `forget_weight` is `F`, `gain_weight` is `G` ([S]: [64, 4608], [2304, 64], [2304, 64]). `middle_cycle_cross_pass_queries` [24, H] holds, for loop layer `j`, the query query at row `2j` and the MLP query at row `2j + 1`; `middle_cycle_cross_pass_kv_queries` [24, H] holds the key query at `2j` and the value query at `2j + 1` ([L] 823-849).

### b) Relative position, clips and scaling

- **No rotary embedding** appears in the text decoder. Attention scale is `head_dim^-0.5` with a head dimension of 128 ([M] 558, 651).
- **Queries** ([M] 458-475): project with `wq_du` [3072, H], split into 24 heads of 128, multiply by a learned per-head scale (`q_l2_clip.scale` [24, 128]), then clip each head vector to L2 norm `qk_l2_clip_threshold` = 22.627 (`x * min(1, t / (norm + 1e-6))`) ([M] 176-202). In global layers only, the clipped query is then multiplied by `tau = 1 + 0.1 * ln(max((pos + 1) / 8192, 1))` ([M] 205-236, 654), where `pos` is the absolute token position ([M] 1327-1330). With `log_scaling_n_floor` = 8192, `tau` is 1 at every position up to and including 8191.
- **Keys** are clipped to the same threshold with no learned scale; **values** are RMS-normalized per head without a weight ([M] 509-513).
- **Relative bias** ([M] 239-270): the query-side input is projected by `wr_du` [384, H] to 24 heads of `d_rel` = 16, scaled per head (`rel_l2_clip.scale` [24, 16]) and clipped to L2 norm 8, log-scaled as above in global layers ([M] 253-257, 664), and multiplied by `rel_logits_proj.proj` [16, 1024] to give, per query position and head, one bias per backward distance 0 to 1023. The bias for a key at distance `d = q_pos - k_pos` ([M] 1249-1268) is that entry, and zero when `d < 0` or `d >= rel_extent` ([M] 268-270). `rel_extent` is the sliding window (1024) in sliding layers and `rel_extent` (1024) in global layers ([M] 557, 650).
- **Logits** are `q . k * 128^-0.5 + bias + mask`, with three query heads per key/value head (head `h` reads key/value head `floor(h / 3)`, [M] 99-108). An extra **sink** logit per head, `attn_sink_proj . query_input` [24, H], joins the softmax with a zero value and only absorbs probability mass; the softmax runs in f32 ([M] 126-137, 619).
- **Output**: `wo_ud (sigmoid(gate_attn . query_input) * attn)`, with `gate_attn` [3072, H] ([M] 531-543, 635).

### c) Block attention residuals

- A block-attention mixture `A(blocks, x)` stacks the block states and the current state, RMS-normalizes each, and attends over them with a learned query, the same normalized states as values, and scale `H^-0.5` ([M] 828-853). Global layers have three queries (query, key, value inputs) and sliding layers one ([M] 1006, 862).
- **Layer body** ([M] 886-910, sliding; 1032-1062, global): with layer input `x`, the attention input is `A(blocks, x)` (layer 0 has none and uses `x`, [M] 862, 1034-1035). The attention output passes through the post-attention convolution. At a boundary layer, `x` is appended to the layer's block states and the residual restarts from the attention output; elsewhere the residual is `x + attention`. The MLP input is `A_mlp(blocks, residual)`, and the layer output is `residual + post_mlp_conv(MoE(RMSNorm(mlp_input)))`.
- `attn_res_normalize_values = true` is the normalization of the values described above ([M] 839-840).
- **Checkpoint names**: `attn_res.attn_queries.N` belongs to layer `N + 1` ([K] 407-411, [L] 1266-1291); [S] gives it shape [3, 2304] at N = 4, 10, 16, 22 (layers 5, 11, 17, 23) and [2304] elsewhere. `attn_res.mlp_queries.N` belongs to layer N; `attn_res.output_query` is the query of `A_out` ([K] 412-415).
- A zero query gives equal logits, so the mixture is the mean of the normalized states. The reference initializes these queries to zero ([M] 1234-1237).

### d) Short convolutions

- **Placement**: on the projected keys and values before their clip and norm ([M] 523-528), on the attention output ([M] 899), and on the MoE output ([M] 909). Kernel size 4, depthwise, no bias, f32.
- **Static form** ([M] 329-393): `y_t = (sum_{i=0..3} w[i] x_{t-3+i} + x_t) / sqrt(5)`. The weight is stored [C, 1, 4] and index 3 multiplies the current token (left padding of k - 1 and the first `seq_len` outputs, [M] 316-322).
- **Dynamic form**, global layers' keys and values ([M] 396-455): taps are per key/value head and per token, `w_t[h][lag] = static_w[lag][c] + (weight_proj . u_t)[h * 4 + lag]` for channel `c` of head `h`, where `u_t` is the projection's input ([M] 430-432, 524-525). `static_w` is stored [4, 1024] and `weight_proj` [32, 2304] ([S]). **Lag 0 is the current token** ([M] 410, 446), the reverse of the static form's index order. `y_t = (sum_{lag=0..3} w_t[h][lag] x_{t-lag} + x_t) / sqrt(9)` ([M] 453-454).
- **Causal state**: the previous three inputs per convolution per cache slot; prefill starts from zeros ([M] 320, 444); one-token decode concatenates the state and the new input ([M] 285-302, 373-376).
- **Sequence and padding masks**: when several sequences are packed into one row, the dynamic form zeroes every tap whose token belongs to another sequence ([M] 448-452), and the static form hands the same sequence index to its kernel ([M] 384). Both forms also zero padded positions before the convolution ([M] 273-282, 367, 436). Lattice runs one sequence per cache with no padding, so every such mask is all ones and lattice applies none. Packed or padded batches are out of scope; supporting them later means adding these masks.

### e) Mixture of experts

- **Router** ([M] 1152-1172), f32: `s = sigmoid(W_r x)` with `W_r` [128, H] (stored as bf16, [S]); select the top 8 by `s + bias` (`mlp.gate.bias` f32 [128], [K] 377); weights are the **unbiased** selected scores divided by their sum plus `1e-20`, times `route_scale` = 8. The bias affects selection only.
- **Expert** ([M] 760-811): `g, u = W13 x` (first half gate, second half up, [M] 809-811); `a = softcap_4(g) * sigmoid(g) * softcap_25(u)`, where the sigmoid sees the uncapped gate ([M] 768-771); `y = W2 a`. Outputs are weighted and accumulated in f32 ([M] 790-805). No shared expert.
- `moe_intermediate_size` is 1152: `W13` is [2304, 2304] and `W2` is [2304, 1152] per expert.

### f) Norms and embeddings

- Sliding layers normalize the attention input with `attn_norm`; global layers have separate `q_input_norm`, `k_input_norm` and `v_input_norm` ([M] 866, 1010-1012; renames [K] 404-406). `mlp_norm` precedes the MoE ([M] 867, 908).
- The embedding row is RMS-normalized with `embed_norm` and **not** multiplied by any scale ([M] 1240-1246).
- Embedding and unembedding are separate tensors, never tied ([M] 1466-1467). Both are stored [201024, H] ([S]). Token ids stay below `unpadded_vocab_size` = 200058, so the padded embedding rows are never read; the reference keeps only the first 200058 unembedding rows ([L] 856-877, [M] 1477).
- Logits are `softcap_30(unembed . h)` in f32 ([M] 1533-1535).

### g) Expert layout in the checkpoint

- **NVFP4**, layers 0 to 19 ([S]): `w13_weight` u8 [128, 2304, 1152] and `w2_weight` u8 [128, 2304, 576], two E2M1 codes per byte with the even element in the low nibble ([L] 777, 803); `*_scale` f8_e4m3 [.., 144] and [.., 72], one scale per 16 consecutive elements of a row; `*_amax` f32 [128] and `*_outer_scale_denominator` f32 [1]. Value = `e2m1 * scale * (amax[e] / denominator)` ([L] 804, 813-814).
- **MXFP8**, layers 20 to 23 ([S]): `w13_weight` f8_e4m3 [128, 2304, 2304], `w2_weight` f8_e4m3 [128, 2304, 1152]; `*_scale` u8 [.., 72] and [.., 36], one power-of-two scale `2^(s - 127)` per 32 consecutive elements of a row ([L] 779, 809-810).
- The scale tensors are row blocks of 16 or 32. The `weight_block_shape` in `hf_quant_config.json` does not describe them.
- **Gate and up are interleaved by row** in `w13`: the loader dequantizes and then de-interleaves dimension 1 ([K] 2658-2666, [L] 232-251), so stored row `2j` is gate row `j` and stored row `2j + 1` is up row `j`. Lattice reads the stored rows in place, each with its own scale row.

## Memory

Derivation from [S] shapes, not a measurement. To be measured when the crate loads the checkpoint.

| Part                                   | Parameters | Stored as          | Bytes       |
| -------------------------------------- | ---------- | ------------------ | ----------- |
| Experts, layers 0-19                   | 20.4 B     | NVFP4 + 1/16 scale | 11.47 GB    |
| Experts, layers 20-23                  | 4.08 B     | MXFP8 + 1/32 scale | 4.20 GB     |
| Embedding and unembedding              | 0.93 B     | bf16               | 1.85 GB     |
| Attention, router, gates, norms, convs | 0.68 B     | bf16               | 1.36 GB     |
| **Total**                              | **26.1 B** |                    | **18.9 GB** |

- The same weights dense would take 52 GB in bf16 and 104 GB in f32, which is why experts stay packed (decision 3).
- Keys and values: 8 heads x 128 x 2 x 4 bytes = 8 KiB per token per cache slot. 40 of the 48 slots are sliding and capped at 1024 tokens (320 MiB together); the 8 global slots add 64 KiB per token, 512 MiB at 8192 tokens.
- A host with 32 GB of memory is the target for the real checkpoint.

## Verification

No reference outputs exist yet. Until they do, the forward is checked on a small synthetic configuration that passes the same configuration validation (a small hidden size, a handful of experts, a window and `rel_extent` of 4, and `log_scaling_n_floor` of 2 so the logarithmic scale is exercised), with seeded weights packed through the same NVFP4 and MXFP8 encoders. These tests are labelled as unverified against the reference. They check:

- each component against a hand computation: the clips and `tau` at chosen positions, the relative-bias gather and its zeroing outside the extent, both convolution tap orders, the sink's absorbed mass, biased selection with unbiased weights, and the softcapped SwiGLU;
- the block and message mixtures with zero queries against the mean of the normalized states, and the message gate with zero weights against the identity;
- a loop with `num_passes = 1` against the same layers run once without the loop;
- incremental decode against full prefill at every position, which exercises the per-pass caches and convolution states.

For the real checkpoint, the bar is the greedy first token on a fixed prompt set, compared with reference logits produced on a host that can run the reference. Logit differences are reported beside the token comparison.

### Parity questions

1. **Weight rounding.** The reference casts dequantized experts to the model dtype, bf16 ([L] 815), so its weights are the decoded values rounded to bf16. Lattice uses the decoded values exactly. A test switch rounds decoded weights to bf16 so the two effects can be told apart.
2. **Activation quantization.** `hf_quant_config.json` names activation quantization, but the reference forward applies none: experts run `F.linear` on dequantized weights ([M] 802-803), and no quantizer is wired into the mapping ([K] 2655-2656). Lattice does not quantize activations.
3. **Activation dtype.** The configuration's dtype is bf16, so the reference rounds activations to bf16 between its f32 regions. Lattice stays in f32. A first-token disagreement or a perplexity gap larger than bf16 rounding can explain points to a structural defect, not to precision.
4. **Long positions.** The logarithmic scale is 1 up to position 8191, so short-prompt parity says nothing about it; the synthetic configuration covers it.
5. **Invalid scales.** The reference multiplies by whatever a scale byte decodes to, including a negative NVFP4 scale and the infinite MXFP8 scale 255. Lattice's NVFP4 decoder refuses a scale with its sign bit set and its E8M0 decoder maps 255 to NaN; the model loader checks every scale tensor once on load and refuses such a checkpoint with an error naming the tensor. A published checkpoint is not expected to contain either.

### Window mask and cache bookkeeping

- **Window**: the configuration maps `sliding_window` to `sliding_window_size` ([C] 90-91), which is 1024 in the published configuration. The sliding mask admits a key when `kv > q - 1024`, on top of the causal mask ([W] 90-99, 132-136, 1196-1201): backward distances 0 to 1023, the same extent as the relative bias ([M] 557, 268-270).
- **Key and value caches**: sliding layers have layer type `hybrid_sliding` and global layers `hybrid` ([C] 150-152), mapped to a cache layer with a sliding window and to one that keeps every token ([Q] 1480-1481, 1994-1995). The sliding cache keeps the last 1023 keys and values and returns them together with the current step's ([Q] 222-249), so a decode step sees exactly distances 0 to 1023.
- **Convolution states**: each of these cache layers holds `number_of_conv_states` = 4 convolution states ([C] 138, [Q] 2002-2003). Prefill stores the last 4 inputs of each convolution, left-padded with zeros when the prompt is shorter; a decode step joins the stored 4 to the new input and stores the last 4 again ([Q] 1244-1282). The forward then reads the last 3 stored inputs and the current one ([M] 444-446). Lattice keeps the previous 3 inputs, the part the forward reads. The decode-against-prefill test checks lattice's own cache against its prefill.

## Alternatives Considered

| Alternative                                          | Why not                                                                                                               |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Dequantize experts to dense bf16 on load             | 52 GB for the weights; excludes a 32 GB host.                                                                         |
| Add the model to `lattice-inference`                 | ADR-103 moves model definitions into model crates behind the facade.                                                  |
| Metal first                                          | Outside the delivery bar; correctness is established on the CPU first, and Metal is measured against it later.        |
| Quantize activations as `hf_quant_config.json` names | The reference does not, so parity would be lost.                                                                      |
| Reuse an existing decoder with options for the loop  | The loop, block mixtures, relative bias and convolutions touch every step of the layer; options would fork it anyway. |

## Consequences

### Positive

- A 26B-parameter mixture-of-experts model runs in about 19 GB on the CPU, through the packed kernels.
- Every behaviour lattice implements is tied to a reference line a reader can check.

### Negative

- One new crate, and 48 cache slots per sequence instead of 24.
- CPU decode of 48 layer applications per token is slow; speed waits for a later Metal path.

### Risks

- Until reference outputs exist, correctness rests on the synthetic tests and on this reading of the reference.
- The window edge and the cache bookkeeping are read from the reference source, not from reference outputs; a mistake there shows first in decode after 1024 tokens.

## References

- Hugging Face transformers at `be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef`: [M], [C], [L], [K], [W], [Q] as tabled above.
- `thinkingmachines/Inkling-2-Mini` on the Hugging Face Hub: `config.json`, `hf_quant_config.json`, and the safetensors header [S].
- ADR-092 (model-owned decoder execution), ADR-100 (foreign quantized checkpoints), ADR-103 (crate decomposition).
