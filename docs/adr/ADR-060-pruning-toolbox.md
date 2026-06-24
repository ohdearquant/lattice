# ADR-060: Pruning Toolbox — Calibration Infrastructure, Structured Pruning, and Analysis Scorers

**Status**: Proposed
**Date**: 2026-05-27
**Crate**: lattice-inference (calibration, scoring, transforms), lattice-tune (recovery, future)
**Research**: RQ-2 (`workspaces/20260527/02.md`)
**Depends on**: ADR-044 (QuaRot rotation infrastructure), ADR-059 (ResidualPolicy, AttentionTag), ADR-061 Phase 1 (CheapOnline metrics substrate — update_ratio and block_influence are the scoring signals ADR-060 consumes)

## Context

Lattice has zero pruning infrastructure. The immediate question is: can we remove 10-50% of a model's parameters without losing quality, and if so, which parameters and by what method?

The RQ-2 survey examined 12 methods across two families. The critical finding for lattice is that **unstructured pruning does not produce inference speedup on Metal** — zeroed weights still flow through dense GEMM. Only **structured pruning** (removing entire layers, FFN neurons, attention groups, or residual dimensions) produces physically smaller tensors that run faster with normal kernels.

A second finding: the claim "50% pruning with less than 1 PPL degradation" is **scale-dependent**. It holds for 13B+ models under SparseGPT/Wanda but not for Qwen3.5-0.8B-class models. STADE's Qwen3 table is the most directly relevant evidence:

| Model      | Dense PPL | 50% STADE PPL | Delta |
| ---------- | --------- | ------------- | ----- |
| Qwen3-32B  | 7.60      | 8.65          | +1.05 |
| Qwen3-14B  | 8.64      | 9.60          | +0.96 |
| Qwen3-8B   | 9.72      | 11.19         | +1.47 |
| Qwen3-1.7B | 16.67     | 18.67         | +2.00 |

Source: STADE (arxiv:2503.22451), Table 4. The trend is clear: smaller models degrade more. For Qwen3.5-0.8B, 50% unstructured pruning will exceed +1 PPL. Conservative structured pruning (10-20%) is the realistic starting point.

Lattice has one critical advantage: **QuaRot's rotation machinery** (ADR-044) shares the same mathematical invariance family as SliceGPT's PCA rotations. Both insert orthogonal rotations into the residual stream and absorb them into neighboring weights. The code infrastructure overlaps; the optimal rotation does not.

### Method Comparison (12 methods surveyed)

| Method                                                                  | Pattern                        | Retraining?                      | Decision cost                    | Representative result                         | Lattice fit                                          |
| ----------------------------------------------------------------------- | ------------------------------ | -------------------------------- | -------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| **Lottery Ticket Hypothesis** (Frankle & Carlin 2019, arxiv:1803.03635) | Unstructured                   | Yes: iterative train-prune-reset | Very high                        | Sparse subnetworks at 10-20% of original      | KG entity only; not implementable for inference-only |
| **SparseGPT** (Frantar & Alistarh 2023, arxiv:2301.00774)               | Unstructured, 2:4, 4:8         | No                               | Hessian inverse, cubic per layer | OPT-175B: 8.35->8.21 at 50%                   | Strong analysis baseline; no Metal speedup           |
| **Wanda** (Sun et al. 2024, arxiv:2306.11695)                           | Unstructured, 2:4, N:M         | No                               | One forward pass                 | LLaMA-30B: 4.77->5.24 at 50%                  | Best first unstructured scorer                       |
| **STADE** (2025, arxiv:2503.22451)                                      | Unstructured, N:M              | No                               | One forward pass                 | Qwen3-14B: 8.64->9.60 at 50%                  | Essential for Qwen/RMSNorm models                    |
| **Wanda++** (2025, arxiv:2503.04992)                                    | Unstructured, N:M              | Regional gradients               | Lightweight                      | 32% PPL improvement over Wanda                | Future Wanda v2; needs gradient infra                |
| **D2Prune** (2026, arxiv:2601.09176)                                    | Unstructured, semi-structured  | Weight update                    | Dual Taylor + attention matching | Improves over SparseGPT/Wanda on Qwen3        | Later research target                                |
| **ShortGPT** (Men et al. 2024, ACL Findings 2025)                       | Layer removal                  | No                               | One forward pass for BI          | LLaMA-2-13B: remove 10/40 layers, MMLU 55->52 | **MVP structured pruner**                            |
| **Gromov Layer Pruning** (2024, arxiv:2403.17887)                       | Contiguous layer blocks        | QLoRA healing                    | Similarity search + PEFT         | Large blocks removable with QLoRA             | After lattice-tune #88                               |
| **Prune&Comp** (2025, arxiv:2507.18212)                                 | Layer removal                  | Training-free compensation       | Iterative pruning + compensation | Frames degradation as magnitude-gap           | Extension after ShortGPT                             |
| **SliceGPT** (Ashkboos et al., ICLR 2024)                               | Residual width                 | No (optional recovery)           | PCA per block, 1024 cal seqs     | OPT-66B: 25% slicing, PPL 9.33->9.68          | **Highest-priority research** (reuses QuaRot)        |
| **LLM-Pruner** (Ma et al. 2023, NeurIPS)                                | Coupled channels/heads/neurons | LoRA recovery                    | Gradient-based importance        | Structural pruning + LoRA in ~3h              | After training loop exists                           |
| **FLAP** (An et al. 2024, AAAI)                                         | Channels/structured            | No                               | One forward pass, fluctuation    | Outperforms LLM-Pruner without retraining     | After activation-stat framework                      |
| **CFSP** (2024, arxiv:2409.13199)                                       | Structured block/channel       | Optional IG-LoRA                 | One forward pass                 | Coarse-to-fine activation info                | Design reference; avoids GQA head pruning            |

## Decision

### D1: Calibration Loop (shared infrastructure — all methods depend on this)

Every pruning method requires: calibration data -> forward pass -> activation capture -> importance scoring. Build this once.

```rust
/// Calibration statistics accumulated during a single forward pass over
/// calibration sequences. One `LayerStats` per transformer layer.
pub struct LayerStats {
    pub layer_idx: usize,
    pub attention_tag: AttentionTag,  // ADR-059 taxonomy (supersedes LayerType)

    // --- Block Influence scoring (ShortGPT) ---
    /// Running dot product: sum_tokens(dot(x_in, x_out))
    pub bi_dot_accumulator: f64,
    /// Running norm product: sum_tokens(||x_in|| * ||x_out||)
    pub bi_norm_accumulator: f64,
    /// Token count for averaging
    pub bi_token_count: u64,

    // --- Per-token activation statistics ---
    /// Per-channel input activation L2 norms (for Wanda scoring)
    /// Shape: [hidden_size]. Accumulated as running sum of squares.
    pub input_channel_sq_norms: Vec<f64>,
    /// Per-channel input activation means (for STADE std-dev scoring)
    pub input_channel_means: Vec<f64>,
    /// Per-channel input activation M2 (Welford online variance)
    pub input_channel_m2: Vec<f64>,

    // --- FFN neuron statistics ---
    /// SwiGLU product |silu(gate_j) * up_j| mean per neuron
    pub ffn_neuron_mean_abs: Vec<f32>,
    /// SwiGLU product RMS per neuron
    pub ffn_neuron_rms: Vec<f32>,
    /// Near-zero fraction per neuron (for dead-neuron detection)
    pub ffn_neuron_near_zero_frac: Vec<f32>,

    // --- Attention head statistics ---
    /// Per-head output norm (for head importance scoring)
    pub head_output_norms: Vec<f32>,

    // --- Covariance accumulator (SliceGPT PCA) ---
    /// Optional: activated only when SliceGPT scoring is requested.
    /// Accumulates C_l = X_l^T X_l for eigendecomposition.
    pub covariance: Option<CovarianceAccumulator>,

    pub token_count: u64,
}

/// Streaming covariance accumulator for SliceGPT PCA.
/// Accumulates X^T X without storing all activations.
pub struct CovarianceAccumulator {
    /// Upper triangle of the covariance matrix, stored as flat vec.
    /// Shape: hidden_size * (hidden_size + 1) / 2
    pub xtx: Vec<f64>,
    pub dim: usize,
    pub sample_count: u64,
}
```

The calibration loop hooks into the forward path at layer boundaries. It does NOT run during normal user inference — it is an offline tool for pruning exploration.

```rust
pub trait CalibrationObserver {
    /// Called at the entry of each transformer block.
    fn observe_block_input(
        &mut self,
        layer: usize,
        attention_tag: AttentionTag,
        x: TensorView<'_>,
    ) -> Result<()>;

    /// Called at the exit of each transformer block (after residual add).
    fn observe_block_output(
        &mut self,
        layer: usize,
        x: TensorView<'_>,
    ) -> Result<()>;

    /// Called with linear projection inputs (for Wanda/STADE per-channel stats).
    fn observe_linear_input(
        &mut self,
        layer: usize,
        projection: ProjectionKind,
        x: TensorView<'_>,
    ) -> Result<()>;

    /// Called with SwiGLU intermediate products (for FFN neuron scoring).
    fn observe_ffn_product(
        &mut self,
        layer: usize,
        gate_silu_times_up: TensorView<'_>,
    ) -> Result<()>;

    /// Finalize statistics and return the accumulated store.
    fn finalize(&mut self) -> Result<Vec<LayerStats>>;
}
```

**How covariance accumulation works for SliceGPT PCA.** For each block `l`, at every calibration token, we receive the residual hidden state `h ∈ R^d`. We accumulate `C_l += h^T h` (outer product, running sum). After all calibration tokens:

1. Normalize: `C_l /= token_count`
2. Eigendecompose: `C_l = Q_l diag(lambda) Q_l^T`
3. Sort eigenvectors by descending eigenvalue
4. `Q_l` is the PCA rotation matrix for block `l`

The streaming accumulator uses O(d^2/2) memory per layer (upper triangle). For Qwen3.5-0.8B with d=1024, that is ~512 KB per layer, ~12 MB total for 24 layers. Acceptable.

### D2: ShortGPT Layer Removal (MVP structured pruner)

**Block Influence formula.** For layer `i`, the Block Influence score is:

```
BI_i = 1 - E_{X,t}[ (X_{i,t}^T X_{i+1,t}) / (||X_{i,t}||_2 ||X_{i+1,t}||_2) ]
```

where `X_{i,t}` is the hidden state for token `t` at the input of layer `i`, and `X_{i+1,t}` is the output after the full block (both sublayers + residual adds). Lower `BI_i` means the layer changes the residual stream less — it is a removal candidate.

**Scoring algorithm** (computed during calibration):

```
for batch in calibration_sequences:
    for layer i in 0..num_layers:
        capture x_in (block input), x_out (block output)
        for token t:
            dot[i] += dot(x_in[t], x_out[t])
            norm[i] += ||x_in[t]|| * ||x_out[t]||

BI[i] = 1 - dot[i] / norm[i]
```

**Qwen3.5 hybrid constraints.** Qwen3.5-0.8B uses a 3:1 hybrid pattern: 18 GatedDeltaNet layers and 6 full-attention layers in groups of `[lin, lin, lin, full] x 6`. A component-ablation study on sub-1B hybrids (arxiv:2603.22473) reports that both component types are essential, linear-attention components can serve as the primary language-modeling backbone, and early layers are disproportionately critical. The safe pruning rules:

```
1. Do NOT prune by layer type alone (don't assume "remove all GDN" or "remove all GQA").
2. Protect early layers by default (protect_first_n = 2 or 4).
3. Require at least one full-attention layer per hybrid group of 4.
4. Score GDN and GQA layers separately — BI may have different distributions.
5. Evaluate ΔPPL after each candidate removal, not just at the end.
```

**Interactive depth-pruning workflow** (the intended user experience):

```
1. Run calibration → compute BI per layer.
2. Rank layers by ascending BI (lowest = most removable).
3. Propose removing the single lowest-BI layer (or a small block).
4. Apply removal in memory (set layer_mask[i] = false in Qwen35Config).
5. Evaluate PPL on a fixed validation slice.
6. If ΔPPL ≤ threshold → accept. Otherwise → rollback (set layer_mask[i] = true).
7. Recompute BI scores on the pruned model (activation distributions shift).
8. Repeat from step 2.
```

**Implementation.** Layer removal uses `Qwen35Config.layer_mask` (already present in `qwen35_config.rs:112-113`). The forward loop checks `layer_mask[i]` and skips the block entirely when false. Runtime metadata updates: `num_hidden_layers` (effective), layer-to-tensor-key mapping, KV cache slot allocation, `layer_types` for hybrid dispatch.

```rust
pub struct ShortGptScorer {
    pub protect_first_n: usize,           // default 4 for Qwen3.5
    pub protect_last_n: usize,            // default 1
    pub min_attention_per_group: usize,   // default 1 (in groups of full_attention_interval)
    pub max_removal_fraction: f32,        // default 0.25
}

impl ImportanceScorer for ShortGptScorer {
    fn score(
        &self,
        model: &Qwen35Config,
        stats: &[LayerStats],
    ) -> Result<Vec<LayerCandidate>> {
        let mut candidates: Vec<LayerCandidate> = stats.iter()
            .enumerate()
            .filter(|(i, _)| {
                *i >= self.protect_first_n
                && *i < model.num_hidden_layers - self.protect_last_n
            })
            .map(|(i, s)| LayerCandidate {
                layer_idx: i,
                attention_tag: s.attention_tag,
                block_influence: s.bi_score(),
            })
            .collect();

        candidates.sort_by(|a, b| a.block_influence.partial_cmp(&b.block_influence).unwrap());

        // Enforce: at least min_attention_per_group full-attention layers
        // in each hybrid group of full_attention_interval layers
        self.enforce_hybrid_group_constraint(model, &mut candidates);

        let max_removable = (model.num_hidden_layers as f32 * self.max_removal_fraction) as usize;
        candidates.truncate(max_removable);
        Ok(candidates)
    }
}
```

### D3: SliceGPT Residual-Width Slicing (high-payoff research path)

**Core idea.** SliceGPT uses the same orthogonal-rotation invariance as QuaRot: insert a rotation into the residual stream, absorb it into neighboring weights, and preserve model function. But the objectives are opposite:

- **QuaRot**: randomized Hadamard rotation to **spread outliers** across dimensions, making int4 quantization work. Uses `BasisKind::HadamardRandomSign`.
- **SliceGPT**: PCA rotation to **concentrate signal energy** into top dimensions, so low-energy dimensions can be deleted. Uses `BasisKind::PcaCalibration`.

They share **code** (the `OrthogonalBasis` trait and rotation-absorption math in `quant::quarot::rotation`), but NOT the same matrix.

**Exact SliceGPT algorithm** (8 steps):

```
Input:
  dense model M with residual width d
  calibration sequences C (1024 sequences, 2048 tokens each)
  slice ratio s (e.g. 0.10, 0.15, 0.25)
  kept width d' = floor((1 - s) * d)

For each block l = 0..num_layers:
  Step 1. Run calibration forward through block l.
          Collect residual states X_l ∈ R^[tokens × d].

  Step 2. Compute covariance: C_l = X_l^T X_l
          (streaming accumulator, no need to store all of X_l)

  Step 3. Eigendecompose C_l = Q_l diag(λ) Q_l^T.

  Step 4. Sort eigenvectors by descending eigenvalue.
          Q_l = [q_1, q_2, ..., q_d] where λ_1 ≥ λ_2 ≥ ... ≥ λ_d.

  Step 5. Fuse Q_l into adjacent weights:
          For each tensor that reads from the residual stream (q_proj, k_proj,
          v_proj, gate_proj, up_proj, in_proj_*):
            W ← W · Q_l^T    (input-side absorption)
          For each tensor that writes to the residual stream (o_proj, down_proj,
          out_proj):
            W ← Q_l · W      (output-side absorption)

  Step 6. Keep only the top d' residual dimensions.
          After descending eigenvalue sort, the first d' coordinates carry
          the most signal energy. Physically slice tensors:
            Input-reading tensors: keep first d' columns (after rotation)
            Output-writing tensors: keep first d' rows (after rotation)

  Step 7. For per-block rotations where Q_{l-1} ≠ Q_l:
          Insert transition matrix Q_{l-1}^T Q_l between blocks.
          This matrix is also sliced: [d'_{l-1} × d'_l].

  Step 8. Evaluate PPL on the sliced model. Accept if ΔPPL ≤ threshold.
```

**Tensor-axis table** (which tensors get sliced on which axis):

| Tensor class                                             | Rotation/slice action                              |
| -------------------------------------------------------- | -------------------------------------------------- |
| Token embedding output axis                              | Rotate and slice residual output axis              |
| `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`     | Rotate/slice the **input residual axis** (columns) |
| `o_proj`, `down_proj`                                    | Rotate/slice the **output residual axis** (rows)   |
| `lm_head`                                                | Rotate/slice the **input residual axis** (columns) |
| GDN `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` | Rotate/slice the **input residual axis** (columns) |
| GDN `out_proj`                                           | Rotate/slice the **output residual axis** (rows)   |
| Residual bridge between differently rotated blocks       | Insert/fuse/slice `Q_{prev}^T Q_next`              |
| Internal attention head dimension                        | Do **not** slice                                   |

**RoPE safety rules.** Baseline SliceGPT must NOT rotate or slice inside the Q/K head coordinate system. RoPE acts after `q_proj` / `k_proj` on per-head Q/K coordinates. If SliceGPT only changes the residual-stream basis and fuses that basis into the **input axes** of projections, RoPE remains unchanged.

```
SAFE:   Slice residual hidden width (d → d').
SAFE:   q_proj becomes [d' → q_heads × head_dim] (nonstandard but implementable).
UNSAFE: Change attention head_dim.
UNSAFE: Delete arbitrary Q/K rotary coordinates.
UNSAFE: Break even/odd RoPE pairs.
```

**OrthogonalBasis trait design** (shared with QuaRot, extended for SliceGPT):

```rust
/// Shared abstraction for orthogonal rotations absorbed into weights.
/// QuaRot uses HadamardRandomSign; SliceGPT uses PcaCalibration.
/// Both use the same absorption math in `quant::quarot::rotation`.
pub trait OrthogonalBasis {
    fn dim(&self) -> usize;

    /// Whether this basis supports the given dimension (power-of-two check
    /// for Hadamard, any dim for PCA/Dense).
    fn supports_dim(&self, n: usize) -> bool;

    /// X ← X Q  (rotate activation vectors right-multiply)
    fn apply(&self, x: &mut [f32], rows: usize, cols: usize) -> Result<()>;

    /// X ← X Q^T  (inverse rotation)
    fn apply_inverse(&self, x: &mut [f32], rows: usize, cols: usize) -> Result<()>;

    /// Absorb rotation into weight matrix input side: W ← W Q^T
    /// Compatible with existing `absorb_input_rotation_f64` in quarot::rotation.
    fn absorb_input_rotation(&self, w: &mut [f64], rows: usize, cols: usize) -> Result<()>;

    /// Absorb rotation into weight matrix output side: W ← Q W
    /// Compatible with existing `absorb_output_rotation_f64` in quarot::rotation.
    fn absorb_output_rotation(&self, w: &mut [f64], rows: usize, cols: usize) -> Result<()>;
}

pub enum BasisKind {
    /// QuaRot (existing, ADR-044): random sign-flip × Walsh-Hadamard.
    /// Spreads outliers across dimensions for uniform int4 quantization.
    HadamardRandomSign { seed: u64 },

    /// SliceGPT (new): PCA eigenvectors from calibration covariance.
    /// Concentrates signal energy into top dimensions for deletion.
    PcaCalibration { eigenvectors: Vec<f64>, eigenvalues: Vec<f64> },

    /// Non-power-of-two support via block-diagonal Hadamard.
    BlockHadamard { block_size: usize, seed: u64 },

    /// Dense orthogonal matrix for debugging / research.
    DenseOrthogonal { matrix: Vec<f64> },
}
```

**Composition order with QuaRot** (SliceGPT first, QuaRot second):

```
dense fp16/bf16 model
  → SliceGPT PCA rotation per block
  → residual-dimension slicing (d → d')
  → evaluate dense pruned model (PPL gate)
  → QuaRot randomized Hadamard rotation in the reduced d' space
  → Q4/Q8 quantization
  → evaluate pruned+quantized model (PPL gate)
```

Do **NOT** apply QuaRot's Hadamard first and then expect PCA slicing to work. Hadamard deliberately spreads information across all coordinates, making dimensions look similarly important. After slicing, the reduced model can be rotated for quantization.

**Implementation detail**: lattice's current `RandomizedHadamard` (ADR-044) requires **power-of-two** dimensions (`hadamard.rs:139`). `d'=896` and `d'=768` are NOT powers of two — they require a `BlockHadamard` basis (block-diagonal Hadamard with block_size being a power of two). The SliceGPT→QuaRot composition is therefore gated on P9's `BlockHadamard` implementation. Until then, the SliceGPT+QuaRot pipeline is restricted to power-of-two `d'` values (e.g., `d'=512` for Qwen3.5-0.8B with d=1024 — a 50% slice, which may be too aggressive). For Qwen3.5-0.8B: d'=896 (12.5% slice) and d'=768 (25%) are the target ratios, but both require `BlockHadamard`.

**Why SliceGPT on Qwen3.5-0.8B is experimental.** The original SliceGPT ICLR paper reports OPT, LLaMA-2, and Phi-2 results; it does NOT provide Qwen-family tables. No published Qwen-family SliceGPT results exist. Treat as research path with strict PPL gates and conservative slice ratios (5-15% for 0.5-1B models).

### D4: Wanda Importance Scoring (analysis tool)

**Scoring formula.** For weight `w_ij` in a linear layer, the Wanda importance score is:

```
score(w_ij) = |w_ij| × ||X_j||_2
```

where `X_j` is the j-th channel of the input activation across all calibration tokens. This captures both weight magnitude and how much the input channel is actually used. One forward pass, no weight updates, no Hessian — the simplest possible unstructured scorer.

**Why Wanda is analysis-only on Metal.** Dense GEMV/GEMM over zeroed weights does not make inference faster. There is no Metal equivalent to NVIDIA Ampere/Hopper Sparse Tensor Cores. Apple's `MPSGraph` has sparse tensor APIs and `Accelerate` has sparse matrix-vector routines, but neither provides drop-in high-throughput sparse LLM GEMM.

```
Dense-with-zeros:      no speedup on Metal.
2:4 sparsity pattern:  no Apple equivalent to NVIDIA Sparse Tensor Cores.
Unstructured sparse:   requires custom CSR/COO kernels (not implemented).
Structured pruning:    real speedup (this is why D2/D3 are prioritized).
```

**STADE fixes Wanda for RMSNorm models.** Wanda's scoring assumes centered input activations (zero-mean), which holds for LayerNorm models but NOT for RMSNorm models like Qwen. RMSNorm normalizes `x / sqrt(mean(x^2) + eps)` without mean subtraction, so activations can be systematically off-center. STADE uses input standard deviation instead of L2 norm, correctly handling the uncentered case. For Qwen-family models, STADE should be the default unstructured scorer once Wanda is implemented.

### D5: GQA Head Pruning (structured, conservative)

**GQA constraints from FlashAttention.** The public FlashAttention interface requires `num_q_heads % num_kv_heads == 0`. This constrains three pruning modes:

| Mode                | What changes                                                  | Risk                                               |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------------- |
| Q-head-only pruning | Remove selected Q heads + corresponding `o_proj` input slices | Must preserve divisibility; creates uneven groups  |
| KV-group pruning    | Remove one KV head and ALL Q heads assigned to it             | Safer shape semantics; larger quality hit per step |
| Mask-only pruning   | Keep tensors, multiply head output by zero                    | No speedup; scoring only                           |

**Lattice default: prune entire GQA groups.** CFSP (arxiv:2409.13199) explicitly avoids pruning attention heads in GQA models because it can cause significant degradation. Lattice follows this constraint for the first deployable version: remove entire KV groups (a KV head plus all its assigned Q heads), not arbitrary Q heads.

**Flash attention compatibility** requires physically smaller tensors:

```
q: [batch, seq, pruned_q_heads, head_dim]
k: [batch, seq, pruned_kv_heads, head_dim]
v: [batch, seq, pruned_kv_heads, head_dim]
Constraint: pruned_q_heads % pruned_kv_heads == 0
```

A mask over heads does NOT make Flash attention faster. Only a physical tensor rewrite can, assuming lattice's Metal attention kernels accept the new head counts.

### D6: SwiGLU FFN Neuron Pruning (structured)

**SwiGLU coupling.** The SwiGLU FFN computes:

```
FFN(x) = W_down · (silu(x W_gate) ⊙ x W_up)
```

Intermediate neuron `j` is coupled across three tensors:

```
gate_proj: output neuron j  (row j of gate_proj.weight)
up_proj:   output neuron j  (row j of up_proj.weight)
down_proj: input column j   (column j of down_proj.weight)
```

Physical neuron pruning must delete all three consistently. Missing any one produces shape mismatches or silent corruption.

**Importance scoring** (activation-only, no gradients needed):

```
score_j = E_tokens[ |silu(g_j) · u_j| ] × ||W_down[j, :]||_2
```

This is a structured, SwiGLU-aware analogue of Wanda's weight-times-activation scoring. The activation contribution `|silu(g_j) · u_j|` measures how much neuron `j` fires; the weight norm `||W_down[j,:]||_2` measures how much that firing affects the output.

An energy variant squares both terms: `E[(silu(g_j) · u_j)^2] × ||W_down[j,:]||_2^2`. Use the energy variant for final ranking; use the linear variant for dead-neuron detection (where absolute magnitude matters more).

**Dead-neuron detection** (streamable during calibration):

```rust
struct FfnNeuronStats {
    count: u64,
    mean_abs: Vec<f32>,     // E[|a_j|] where a_j = silu(g_j) * u_j
    rms: Vec<f32>,          // sqrt(E[a_j^2])
    max_abs: Vec<f32>,      // max over calibration tokens
    near_zero_fraction: Vec<f32>,  // fraction of tokens where |a_j| < epsilon
}
```

A neuron is a dead-neuron candidate if `rms < ε_rms AND max_abs < ε_max AND near_zero_fraction > threshold`. Do not prune solely on deadness unless the calibration set is representative.

**Conservative targets** (without retraining):

```
0.5-1B models:   5-15% FFN neurons
7B-14B models:  10-30%
30B+ models:    20-40%, with PPL verification per step
```

### D7: PPL Gate (mandatory for all operations)

Every prune operation evaluates PPL before/after on a fixed validation corpus. Accept only if `ΔPPL ≤ threshold`:

```rust
pub struct PplGate {
    /// Maximum acceptable PPL increase. Conservative: 0.3; aggressive: 1.0.
    pub max_delta_ppl: f32,
    /// Validation corpus path (e.g. WikiText-2 raw test split).
    pub validation_corpus: PathBuf,
    /// Number of tokens to evaluate (0 = full corpus).
    pub max_eval_tokens: usize,
    /// If true, automatically rollback the prune when gate fails.
    pub rollback_on_fail: bool,
}

pub struct PplResult {
    pub dense_ppl: f32,
    pub pruned_ppl: f32,
    pub delta_ppl: f32,
    pub tokens_scored: usize,
    pub passed: bool,
}
```

The PPL evaluator reuses the existing strided sliding-window perplexity harness from ADR-044 step 4 (`qwen35::eval::run_strided_perplexity`).

### D8: PrunePlan data structure

```rust
/// Complete specification of a pruning decision.
/// JSON-serializable for checkpoint provenance and reproducibility.
#[derive(Serialize, Deserialize)]
pub struct PrunePlan {
    pub method: PruneMethod,
    pub source_model: ModelChecksum,

    pub calibration: CalibrationConfig,

    /// Layers removed (ShortGPT).
    pub removed_layers: Vec<usize>,
    /// FFN neurons kept per layer (structured FFN pruning).
    pub ffn_keep_indices: BTreeMap<usize, Vec<usize>>,
    /// Attention groups kept per layer (GQA group pruning).
    pub attention_group_keep_indices: BTreeMap<usize, Vec<usize>>,
    /// Residual width after slicing (SliceGPT). None = no slicing.
    pub residual_width: Option<usize>,
    /// PCA rotation matrices per block (SliceGPT). None = no rotation.
    pub pca_rotations: Option<Vec<PcaRotationSpec>>,

    pub constraints: PruneConstraints,
    pub metrics: Option<PplResult>,
}

pub struct PruneConstraints {
    pub preserve_rope_pairs: bool,
    pub preserve_gqa_grouping: bool,
    pub min_full_attention_layers_per_group: usize,
    pub protect_first_n_layers: usize,
    pub protect_last_n_layers: usize,
    pub hidden_dim_multiple: usize,     // for SliceGPT: d' must be multiple of this
    pub ffn_dim_multiple: usize,        // for FFN pruning: intermediate_size divisibility
    pub max_delta_ppl: Option<f32>,
}
```

### D9: Quantitative Bounds — What Meets ">=30% Reduction and <=1 PPL Delta"?

The strict answer is scale-dependent.

**Unstructured weight pruning** has the strongest PPL numbers. SparseGPT on OPT-175B: 50% sparsity, PPL 8.35->8.21 (improves!). Wanda on LLaMA-30B: 50%, PPL 4.77->5.24 (+0.47). But Wanda on LLaMA-7B: 50%, PPL 5.68->7.26 (+1.58, fails the gate).

**Structured pruning** is more deployment-friendly but harder to keep within +1 PPL at aggressive ratios. SliceGPT OPT-66B at 25%: PPL 9.33->9.68 (+0.35, passes). LLaMA-2-70B at 25%: PPL 3.32->4.60 (+1.28, fails).

**For Qwen3.5-0.8B specifically**: no method should promise 50% pruning with less than 1 PPL degradation. The realistic starting point is 10-20% structured pruning with strict PPL gates. The iterate-and-measure workflow is more important than any single method.

## Alternatives Considered

| Alternative                          | Pros                     | Cons                                         | Why not                                        |
| ------------------------------------ | ------------------------ | -------------------------------------------- | ---------------------------------------------- |
| SparseGPT first                      | Best quality             | Hessian inverse, cubic per layer, complex    | Wanda gives 80% of analysis value at 5% cost   |
| Unstructured + sparse kernels        | Real speedup             | Metal has no sparse GEMM                     | Infrastructure doesn't exist                   |
| LLM-Pruner (gradient-based)          | Better importance scores | Requires backward pass                       | No training loop yet                           |
| Skip calibration, use magnitude-only | Simpler                  | Magnitude alone misses activation importance | Wanda proved activation-weight product matters |

## Consequences

### Positive

- **SliceGPT + QuaRot composition** is lattice's differentiator: prune residual dimensions (SliceGPT PCA) -> quantize remaining dimensions (QuaRot Hadamard) -> smallest possible dense model with quality gates. No other pure-Rust engine has this pipeline.
- **Calibration loop** serves double duty: pruning importance scores and metrics infrastructure.
- **Incremental pruning** is naturally supported by the iterate/measure/rollback design.
- The entire pruning toolbox produces standard dense SafeTensors — no special runtime needed.

### Negative

- SliceGPT PCA on Qwen3.5-0.8B is experimental — no published Qwen-family results exist. Must treat as research path.
- Unstructured pruning (Wanda/STADE) gives no Metal speedup. Value is analysis-only until sparse kernels exist.
- Calibration loop adds ~12 MB memory for SliceGPT covariance accumulators (24 layers × 512 KB each).
- Per-block PCA eigendecomposition on d=1024 requires O(d^3) ~ 10^9 FLOPs per layer. Acceptable offline.

### Risks

- **Hybrid layer removal fragility.** Removing the wrong layer from a Qwen3.5 hybrid model (especially an early layer or the last full-attention layer in a group) could disproportionately degrade quality. Mitigation: strict protect_first_n, min_attention_per_group constraints, and per-step PPL verification.
- **SliceGPT transition matrices.** Per-block PCA rotations require transition matrices `Q_{l-1}^T Q_l` between blocks. If rotations differ significantly between adjacent blocks, this matrix is not sparse and adds overhead. Mitigation: start with a single global PCA rotation (same Q for all blocks) before implementing per-block rotations.
- **RoPE pair invariant violation.** Any pruning that accidentally breaks the even/odd pairing of RoPE coordinates will produce wrong position-dependent attention scores. Mitigation: `PruneConstraints.preserve_rope_pairs = true` by default; validate after any head-dimension-touching operation.
- **LoRA incompatibility with rotated models.** Same issue as QuaRot (ADR-044 Risks): rotated base weights operate in a different basis than un-rotated adapter weights. A SliceGPT-pruned model needs adapter weights projected into the reduced PCA basis. Deferred to post-lattice-tune era.

## Implementation Plan

| Phase | Scope                                                                | Files                                                                                              | Depends on                         | Estimated PR  |
| ----- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------- |
| P0    | `CalibrationObserver` trait + `LayerStats` + forward hook points     | `crates/inference/src/prune/calibration.rs`, `crates/inference/src/prune/mod.rs`, forward.rs hooks | --                                 | Single PR     |
| P1    | `CovarianceAccumulator` (streaming X^T X)                            | `crates/inference/src/prune/covariance.rs`                                                         | P0                                 | Same PR as P0 |
| P2    | `BlockInfluenceScorer` + `ShortGptScorer`                            | `crates/inference/src/prune/scoring/block_influence.rs`                                            | P0                                 |               |
| P3    | `LayerRemovalTransform` (apply layer_mask, update config)            | `crates/inference/src/prune/transforms/remove_layers.rs`                                           | P2, ADR-059 `ResidualPolicy::Skip` |               |
| P4    | `PplGate` (reuse `run_strided_perplexity`)                           | `crates/inference/src/prune/eval.rs`                                                               | P0, ADR-044 step 4 harness         |               |
| P5    | `PrunePlan` serialization + `lattice_pruning.json` metadata          | `crates/inference/src/prune/plan.rs`                                                               | P0                                 |               |
| P6    | `WandaScorer` + `StadeScorer` + mask export                          | `crates/inference/src/prune/scoring/wanda.rs`, `stade.rs`                                          | P0                                 |               |
| P7    | `FfnSwiGluScorer` + `FfnNeuronPruneTransform`                        | `crates/inference/src/prune/scoring/ffn_swiglu.rs`, `transforms/prune_ffn.rs`                      | P0                                 |               |
| P8    | `GqaGroupScorer` + `GqaGroupPruneTransform` + Flash shape validation | `crates/inference/src/prune/scoring/attention_heads.rs`, `transforms/prune_heads.rs`               | P0                                 |               |
| P9    | `OrthogonalBasis` trait refactor from `quant::quarot`                | `crates/inference/src/quant/orthogonal_basis.rs`                                                   | ADR-044 existing code              |               |
| P10   | `PcaCalibration` basis + `SliceGptScorer`                            | `crates/inference/src/prune/scoring/slice_pca.rs`                                                  | P1, P9                             |               |
| P11   | `SliceResidualTransform` (tensor rewriting)                          | `crates/inference/src/prune/transforms/slice_residual.rs`                                          | P10                                |               |
| P12   | SliceGPT -> QuaRot composition pipeline                              | `crates/inference/src/prune/pipeline.rs`                                                           | P11, ADR-044                       |               |
| P13   | LoRA/QLoRA recovery after pruning                                    | `crates/tune/src/prune_recovery.rs`                                                                | lattice-tune #88                   |               |

**Dependency ordering for crate publishing**: all pruning code lives in `lattice-inference` (existing crate). No new crate needed. P13 touches `lattice-tune` (separate crate, ships later).

**SafeTensors export layout** for pruned models:

```
model.safetensors          # physically smaller tensors
config.json                # updated num_hidden_layers, hidden_size, etc.
lattice_pruning.json       # full provenance: method, calibration, plan, metrics
```

For unstructured masks (analysis export only):

```
research_masks.safetensors  # binary masks per tensor
lattice_pruning.json        # method, sparsity stats
```

## References

### Papers

- SparseGPT: Frantar & Alistarh 2023, [arxiv:2301.00774](https://arxiv.org/abs/2301.00774)
- Wanda: Sun et al. 2024, [arxiv:2306.11695](https://arxiv.org/abs/2306.11695)
- STADE: 2025, [arxiv:2503.22451](https://arxiv.org/html/2503.22451v2)
- SliceGPT: Ashkboos et al. ICLR 2024, [OpenReview](https://openreview.net/forum?id=vXxardq6db)
- ShortGPT: Men et al. 2024, [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1035.pdf)
- Gromov Layer Pruning: 2024, [arxiv:2403.17887](https://arxiv.org/abs/2403.17887)
- LLM-Pruner: Ma et al. NeurIPS 2023, [OpenReview](https://openreview.net/forum?id=J8Ajf9WfXP)
- FLAP: An et al. AAAI 2024, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28960)
- CFSP: 2024, [arxiv:2409.13199](https://arxiv.org/html/2409.13199v2)
- Wanda++: 2025, [arxiv:2503.04992](https://arxiv.org/abs/2503.04992)
- D2Prune: 2026, [arxiv:2601.09176](https://arxiv.org/abs/2601.09176)
- Prune&Comp: 2025, [arxiv:2507.18212](https://arxiv.org/html/2507.18212v1)
- Lottery Ticket Hypothesis: Frankle & Carlin 2019, [arxiv:1803.03635](https://arxiv.org/abs/1803.03635)
- QuaRot: Ashkboos et al. NeurIPS 2024, [arxiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Hybrid component ablation: 2026, [arxiv:2603.22473](https://arxiv.org/abs/2603.22473)
- Layer relevance beyond cosine: 2026, [arxiv:2605.14075](https://arxiv.org/abs/2605.14075)
- Voita head pruning: 2019, [arxiv:1905.10650](https://arxiv.org/abs/1905.10650)
- Free Lunch LLM Compression: 2026, [OpenReview](https://openreview.net/forum?id=PaMj3yuaHi)

### Lattice Code

- QuaRot infrastructure: `crates/inference/src/quant/quarot/` (ADR-044)
- Rotation absorption: `quant::quarot::rotation` (input-side and output-side)
- Rotation plan: `quant::quarot::plan` (per-tensor recipe for Qwen3.5 hybrid)
- Qwen3.5 config: `crates/inference/src/model/qwen35_config.rs` (AttentionTag via ADR-059, layer_mask field)
- PPL evaluator: `qwen35::eval::run_strided_perplexity` (ADR-044 step 4)
- ResidualPolicy: ADR-059

### KG Entities

- Structured Pruning: `02a37d6f`
- Unstructured Pruning: `fe89bb23`
- SparseGPT: `cacd4896`
- Wanda: `f2ccb670`
- STADE: `0d6668a5`
- SliceGPT: `95792865`
- ShortGPT: `28584ee2`
- Block Influence: `d6e7c165`
- Gromov Layer Pruning: `4ada1f72`
- LLM-Pruner: `74b3b6dc`
- FLAP: `eea854d1`
- CFSP: `b58504ec`
- Wanda++: `27ffd2ee`
- D2Prune: `71395373`
- Prune and Comp: `8fd85ff6`
- Calibration Loop: `e742ab58`
- PPL Gate: `e975ea36`
- PrunePlan: `37aab1d0`
- OrthogonalBasis Trait: `f227231c`
- PcaCalibration: `e09b5b6c`
- SwiGLU FFN Pruning: `c820e507`
- GQA Head Pruning: `261d69e2`
- N:M Sparsity: `8b573a58`
- 2:4 Sparsity: `5d764448`
- Lottery Ticket Hypothesis: `6b08bfba`
- QuaRot (existing): `e754741e`
- lattice-inference (existing): `6c0a97df`

## Implementation status (2026-06-24)

Only the D2 ShortGPT block-influence scorer has shipped. `BlockInfluence` and
`BlockInfluenceAccumulator` are implemented at `crates/inference/src/pruning.rs:62` and `121`.
The `CalibrationObserver` trait (D1/P0), `Wanda` per-channel activation scorer, and `SliceGPT`
PCA-rotation infrastructure are not present in source — `pruning.rs` module doc notes that
`CalibrationObserver` and `ForwardCtx` hooks are "ADR-060 D1/P0 work (future PR)". Grep for
`Wanda`, `SliceGPT`, and `CalibrationObserver` in `crates/` returns no type definitions. The
`OrthogonalBasis` trait and `BasisKind` enum referenced in the ADR-044 Amendment are also not
yet implemented (they are a design proposal).
