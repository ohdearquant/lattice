//! Metal port of the Qwen3.5-0.8B ViT forward pass (ADR-069 S3b), gated
//! against the S3a CPU reference ([`super::qwen35_vit::qwen35_vit_forward`]).
//!
//! **Zero independent convention decisions**: every layout/normalization/
//! activation/RoPE convention here is inherited directly from the CPU
//! reference — the setup tables (bilinear position-embedding interpolation,
//! 2-axis vision RoPE `cos`/`sin` construction) are computed by literally
//! calling [`super::qwen35_vit::build_pos_embed_and_rope_tables`], and the
//! per-element ops (biased `LayerNorm`, GELU-tanh, RoPE rotate-half
//! application, fail-closed softmax) reuse the exact CPU reference functions
//! (`layer_norm`, `gelu`, `apply_rope_inplace`, `softmax_inplace`) rather
//! than re-deriving them. Only the GEMM-heavy compute — patch embedding,
//! fused QKV projection, the two attention matmuls (`Q @ K^T` and
//! `scores @ V`), the output projection, and the MLP `fc1`/`fc2` — is
//! dispatched to the Metal GPU via [`crate::forward::metal_gemm`]'s existing
//! `metal_matmul` / `metal_matmul_bt` kernels (reused as-is, not
//! reimplemented — these already carry the codebase's tiled-GEMM Metal
//! shader and CPU-availability fallback). Those GEMMs account for well over
//! 99% of the forward pass's FLOPs at the real depth-12/hidden-768/12-head
//! geometry; the reused elementwise CPU functions are the strongest form of
//! convention mirroring available (literally the same code, not a
//! re-implementation), and their cost is negligible relative to the GEMMs.
//!
//! Gate: `tests/vision_s3b_vit_metal_gate_test.rs` — cosine > 0.999 between
//! this forward and the S3a CPU reference on the same (synthetic,
//! deterministic) weights and the committed golden fixture image, under the
//! machine-wide `gpu_test_lock()`.

#[allow(unused_imports)] // only used by the non-metal-gpu stub below
use super::VisionError;
#[allow(unused_imports)]
use super::checkpoint::Qwen35VisionWeights;
#[allow(unused_imports)]
use super::qwen35_vit::GridThw;
#[allow(unused_imports)]
use crate::model::qwen35_config::VisionModelConfig;

// Real implementation lives behind the same `metal-gpu` feature gating as
// the rest of the codebase's Metal code (`metal_gemm.rs`'s `mod gpu { .. }`
// pattern, reused here). A stub below keeps the symbol callable — returning
// a clear error — from builds where the feature is off, so callers (and the
// S3b gate test) don't need to sprinkle `cfg` at every call site.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod gpu {
    use super::super::VisionError;
    use super::super::checkpoint::Qwen35VisionWeights;
    use super::super::qwen35_vit::GridThw;
    use super::super::qwen35_vit::{apply_rope_inplace, build_pos_embed_and_rope_tables};
    use super::super::vit::{gelu, layer_norm, softmax_inplace};
    use crate::forward::metal_gemm::{metal_matmul, metal_matmul_bt};
    use crate::model::qwen35_config::VisionModelConfig;
    #[cfg(feature = "test-utils")]
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Test-only diagnostic (issue: S3b gate review — cosine parity alone
    /// can't distinguish "every GEMM ran on Metal" from "every GEMM silently
    /// fell back to the CPU loop below", since the fallback is exact). Counts
    /// GEMM calls that actually dispatched to the GPU (`metal_matmul`/
    /// `metal_matmul_bt` returned `true`), not calls that took the CPU
    /// fallback branch. Compiled only under the non-default `test-utils`
    /// feature (see `Cargo.toml`), so the normal production `metal-gpu` build
    /// carries neither the storage nor the `fetch_add` at each dispatch site
    /// — unlike the opt-in-flag `PathProofCounters` pattern in
    /// `forward/metal_qwen35.rs` (issue #239), which stays compiled in and
    /// gates the atomic op behind a runtime check instead. Read via
    /// [`metal_dispatch_count`] / zeroed via [`reset_metal_dispatch_count`];
    /// see `tests/vision_s3b_vit_metal_gate_test.rs`.
    #[cfg(feature = "test-utils")]
    static METAL_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

    /// Returns the number of Metal GEMM calls dispatched to the GPU (as
    /// opposed to the CPU fallback branch) since the last
    /// [`reset_metal_dispatch_count`] call. Test-only diagnostic, compiled
    /// only under the `test-utils` feature.
    #[cfg(feature = "test-utils")]
    pub fn metal_dispatch_count() -> u64 {
        METAL_DISPATCH_COUNT.load(Ordering::Relaxed)
    }

    /// Resets the Metal GEMM dispatch counter to zero. Test-only diagnostic,
    /// compiled only under the `test-utils` feature.
    #[cfg(feature = "test-utils")]
    pub fn reset_metal_dispatch_count() {
        METAL_DISPATCH_COUNT.store(0, Ordering::Relaxed);
    }

    /// `C[m, n] = A[m, k] @ B[n, k]^T`, dispatched to Metal when the codebase's
    /// existing `metal_matmul_bt` accepts the shape (real ViT geometry always
    /// clears its GPU-dispatch-threshold); falls back to the identical CPU dot
    /// product otherwise, so the result is correct regardless of GPU
    /// availability. This is the exact math [`super::qwen35_vit::qwen35_vit`]'s
    /// `batch_matvec` performs (`A` = activations `[m, k]`, `B` = weight
    /// `[n, k]` row-major, i.e. PyTorch `nn.Linear` layout).
    fn gemm_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        if metal_matmul_bt(a, b, &mut c, m, k, n) {
            #[cfg(feature = "test-utils")]
            METAL_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            for i in 0..m {
                let ai = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let bj = &b[j * k..(j + 1) * k];
                    let mut acc = 0.0f32;
                    for t in 0..k {
                        acc += ai[t] * bj[t];
                    }
                    c[i * n + j] = acc;
                }
            }
        }
        c
    }

    /// `C[m, n] = A[m, k] @ B[k, n]`, dispatched to Metal via `metal_matmul`;
    /// same CPU-fallback contract and dispatch-counting as [`gemm_bt`].
    fn gemm_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        if metal_matmul(a, b, &mut c, m, k, n) {
            #[cfg(feature = "test-utils")]
            METAL_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        } else {
            for i in 0..m {
                let ai = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for t in 0..k {
                        acc += ai[t] * b[t * n + j];
                    }
                    c[i * n + j] = acc;
                }
            }
        }
        c
    }

    /// Full (unwindowed) self-attention, mirroring
    /// [`super::qwen35_vit::qwen35_vit`]'s `multihead_attention_full` exactly
    /// (same per-head Q/K/V extraction, same `scale` application order, same
    /// fail-closed row softmax reused directly), but with the two `[n, n]`- and
    /// `[n, head_dim]`-shaped matmuls per head dispatched to Metal.
    fn multihead_attention_full_metal(
        qkv: &[f32],
        n: usize,
        hidden: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * hidden];

        for h in 0..n_heads {
            let mut q_h = vec![0.0f32; n * head_dim];
            let mut k_h = vec![0.0f32; n * head_dim];
            let mut v_h = vec![0.0f32; n * head_dim];
            for i in 0..n {
                let base = i * 3 * hidden;
                q_h[i * head_dim..(i + 1) * head_dim]
                    .copy_from_slice(&qkv[base + h * head_dim..base + (h + 1) * head_dim]);
                k_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                    &qkv[base + hidden + h * head_dim..base + hidden + (h + 1) * head_dim],
                );
                v_h[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                    &qkv[base + 2 * hidden + h * head_dim..base + 2 * hidden + (h + 1) * head_dim],
                );
            }

            // scores[n, n] = Q_h @ K_h^T, then scale (same order as the CPU
            // reference: dot product first, `* scale` applied to the raw dot).
            let mut scores = gemm_bt(&q_h, &k_h, n, head_dim, n);
            for s in scores.iter_mut() {
                *s *= scale;
            }
            for i in 0..n {
                softmax_inplace(&mut scores[i * n..(i + 1) * n]);
            }

            // out_h[n, head_dim] = scores @ V_h
            let out_h = gemm_nn(&scores, &v_h, n, n, head_dim);
            for i in 0..n {
                out[i * hidden + h * head_dim..i * hidden + (h + 1) * head_dim]
                    .copy_from_slice(&out_h[i * head_dim..(i + 1) * head_dim]);
            }
        }

        out
    }

    /// Metal port of [`super::qwen35_vit::qwen35_vit_forward`]. Same signature,
    /// same output contract (`[num_patches, hidden_size]` pre-merger hidden
    /// states, no post-block normalization). See module docs for the
    /// GPU/CPU-reuse split.
    ///
    /// # Errors
    ///
    /// [`VisionError::ShapeMismatch`] if `pixel_values.len()` doesn't match
    /// `grid.num_patches() * (in_channels * temporal_patch_size * patch_size^2)`.
    pub fn qwen35_vit_forward_metal(
        weights: &Qwen35VisionWeights,
        cfg: &VisionModelConfig,
        pixel_values: &[f32],
        grid: GridThw,
    ) -> Result<Vec<f32>, VisionError> {
        let hidden = cfg.hidden_size;
        let n = grid.num_patches();
        let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        if pixel_values.len() != n * patch_len {
            return Err(VisionError::ShapeMismatch {
                expected: n * patch_len,
                actual: pixel_values.len(),
                context: "qwen35_vit_forward_metal: pixel_values length".into(),
            });
        }

        // ---- Patch embedding (Metal GEMM). ----
        let mut hidden_states = gemm_bt(
            pixel_values,
            &weights.patch_embed_weight,
            n,
            patch_len,
            hidden,
        );
        for i in 0..n {
            for j in 0..hidden {
                hidden_states[i * hidden + j] += weights.patch_embed_bias[j];
            }
        }

        // ---- Position embedding + RoPE tables: exact CPU-reference setup reuse. ----
        let head_dim = hidden / cfg.num_heads;
        let (pos_embed_contrib, cos_table, sin_table) =
            build_pos_embed_and_rope_tables(weights, cfg, grid);
        for i in 0..n * hidden {
            hidden_states[i] += pos_embed_contrib[i];
        }

        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let n_heads = cfg.num_heads;

        for block in &weights.blocks {
            // -- Attention sub-layer --
            let residual = hidden_states.clone();
            let mut normed = hidden_states.clone();
            for i in 0..n {
                layer_norm(
                    &mut normed[i * hidden..(i + 1) * hidden],
                    &block.norm1_weight,
                    &block.norm1_bias,
                    1e-6,
                );
            }

            let mut qkv = gemm_bt(&normed, &block.qkv_weight, n, hidden, 3 * hidden);
            for i in 0..n {
                for j in 0..3 * hidden {
                    qkv[i * 3 * hidden + j] += block.qkv_bias[j];
                }
            }

            // Apply RoPE to Q and K in place, per head — exact CPU-reference
            // rotate-half function, reused directly (not re-derived).
            for i in 0..n {
                let base = i * 3 * hidden;
                let cos_row = &cos_table[i * head_dim..(i + 1) * head_dim];
                let sin_row = &sin_table[i * head_dim..(i + 1) * head_dim];
                for h in 0..n_heads {
                    let q = &mut qkv[base + h * head_dim..base + (h + 1) * head_dim];
                    apply_rope_inplace(q, cos_row, sin_row);
                    let k_base = base + hidden;
                    let k = &mut qkv[k_base + h * head_dim..k_base + (h + 1) * head_dim];
                    apply_rope_inplace(k, cos_row, sin_row);
                }
            }

            let attn_out =
                multihead_attention_full_metal(&qkv, n, hidden, n_heads, head_dim, scale);
            let proj_out = gemm_bt(&attn_out, &block.proj_weight, n, hidden, hidden);
            for i in 0..n * hidden {
                hidden_states[i] = residual[i] + proj_out[i] + block.proj_bias[i % hidden];
            }

            // -- MLP sub-layer --
            let residual = hidden_states.clone();
            let mut normed = hidden_states.clone();
            for i in 0..n {
                layer_norm(
                    &mut normed[i * hidden..(i + 1) * hidden],
                    &block.norm2_weight,
                    &block.norm2_bias,
                    1e-6,
                );
            }

            let mlp_dim = block.fc1_bias.len();
            let mut fc1_out = gemm_bt(&normed, &block.fc1_weight, n, hidden, mlp_dim);
            for i in 0..n {
                for j in 0..mlp_dim {
                    let idx = i * mlp_dim + j;
                    fc1_out[idx] = gelu(fc1_out[idx] + block.fc1_bias[j]);
                }
            }
            let fc2_out = gemm_bt(&fc1_out, &block.fc2_weight, n, mlp_dim, hidden);
            for i in 0..n * hidden {
                hidden_states[i] = residual[i] + fc2_out[i] + block.fc2_bias[i % hidden];
            }
        }

        Ok(hidden_states)
    }
} // mod gpu

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use gpu::qwen35_vit_forward_metal;

/// Test-only diagnostic re-exports — see [`gpu::metal_dispatch_count`] docs.
/// Gated on `test-utils` in addition to `metal-gpu` so the normal production
/// build carries neither the counter storage nor its `fetch_add` sites.
#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "test-utils"))]
pub use gpu::{metal_dispatch_count, reset_metal_dispatch_count};

/// Stub for builds without the `metal-gpu` feature (or off macOS): returns a
/// clear error instead of silently running a CPU-only re-implementation, so
/// callers can tell "Metal unavailable" apart from "Metal ran and matched".
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn qwen35_vit_forward_metal(
    _weights: &Qwen35VisionWeights,
    _cfg: &VisionModelConfig,
    _pixel_values: &[f32],
    _grid: GridThw,
) -> Result<Vec<f32>, VisionError> {
    Err(VisionError::InvalidConfig(
        "qwen35_vit_forward_metal requires the `metal-gpu` feature on macOS".into(),
    ))
}

#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
mod tests {
    use super::gpu::qwen35_vit_forward_metal;
    use super::*;
    use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
    use crate::vision::qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward};

    fn tiny_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            temporal_patch_size: 1,
            num_position_embeddings: 16,
            in_channels: 1,
            deepstack_visual_indexes: vec![],
        }
    }

    fn make_test_png(w: u32, h: u32) -> Vec<u8> {
        use image::RgbImage;
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x + y) % 256) as u8;
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn make_test_weights(cfg: &VisionModelConfig) -> Qwen35VisionWeights {
        let hidden = cfg.hidden_size;
        let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let mlp_dim = 2 * hidden;
        let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;

        let mut state = 0x1234_5678_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 0.2 - 0.1
        };
        let mut v = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();

        let block = VisualBlockWeights {
            qkv_weight: v(3 * hidden * hidden),
            qkv_bias: v(3 * hidden),
            proj_weight: v(hidden * hidden),
            proj_bias: v(hidden),
            fc1_weight: v(mlp_dim * hidden),
            fc1_bias: v(mlp_dim),
            fc2_weight: v(hidden * mlp_dim),
            fc2_bias: v(hidden),
            norm1_weight: vec![1.0; hidden],
            norm1_bias: vec![0.0; hidden],
            norm2_weight: vec![1.0; hidden],
            norm2_bias: vec![0.0; hidden],
        };

        Qwen35VisionWeights {
            patch_embed_weight: v(hidden * patch_len),
            patch_embed_weight_shape: vec![
                hidden,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ],
            patch_embed_bias: v(hidden),
            pos_embed: v(cfg.num_position_embeddings * hidden),
            blocks: vec![block],
            merger: VisualMergerWeights {
                fc1_weight: v(merge_in * merge_in),
                fc1_bias: v(merge_in),
                fc2_weight: v(cfg.out_hidden_size * merge_in),
                fc2_bias: v(cfg.out_hidden_size),
                norm_weight: vec![1.0; hidden],
                norm_bias: vec![0.0; hidden],
            },
        }
    }

    /// Below-GPU-dispatch-threshold shapes exercise the CPU-fallback path in
    /// `gemm_bt`/`gemm_nn` on every platform (no Metal device required),
    /// proving the fallback math matches the CPU reference exactly.
    #[test]
    fn metal_forward_matches_cpu_reference_small_shapes() {
        let cfg = tiny_cfg();
        let weights = make_test_weights(&cfg);
        let png = make_test_png(8, 8);
        let (pixel_values, grid) = preprocess_qwen35_image(&png, &cfg, None).expect("preprocess");

        let cpu_out = qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("cpu forward");
        let metal_out =
            qwen35_vit_forward_metal(&weights, &cfg, &pixel_values, grid).expect("metal forward");

        assert_eq!(cpu_out.len(), metal_out.len());
        for (a, b) in cpu_out.iter().zip(metal_out.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "cpu={a} metal={b} diverge beyond fallback-path tolerance"
            );
        }
    }

    #[test]
    fn metal_forward_rejects_pixel_length_mismatch() {
        let cfg = tiny_cfg();
        let weights = make_test_weights(&cfg);
        let grid = GridThw { t: 1, h: 4, w: 4 };
        let bad_pixels = vec![0.0f32; 3];
        let err = qwen35_vit_forward_metal(&weights, &cfg, &bad_pixels, grid).unwrap_err();
        assert!(matches!(err, VisionError::ShapeMismatch { .. }));
    }
}
