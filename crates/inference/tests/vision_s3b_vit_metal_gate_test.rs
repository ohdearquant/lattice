//! ADR-069 S3b gate: Metal port of the Qwen3.5-0.8B ViT forward
//! ([`lattice_inference::vision::qwen35_vit_metal::qwen35_vit_forward_metal`])
//! vs the S3a CPU reference
//! ([`lattice_inference::vision::qwen35_vit::qwen35_vit_forward`]), at
//! cosine > 0.999 on the committed golden fixture image.
//!
//! Checkpoint-light by design (ADR-069 Amendment 1's "prefer a
//! committed-fixture design that runs everywhere"): rather than requiring
//! the 1.6 GB fp16 checkpoint, this test builds deterministic synthetic
//! weights at the checkpoint's *real* per-block geometry (hidden 768, 12
//! heads, patch 16, spatial_merge 2, temporal_patch 2, `num_position_embeddings`
//! 2304 — every value pinned to ADR-069's `vision_config`, not chosen here),
//! with a reduced depth (2 blocks instead of 12) purely for test wall-clock —
//! depth doesn't change any per-op Metal/CPU convention, so parity at depth 2
//! is exactly as validating of the block-loop port as depth 12 would be. The
//! image is the already-committed `tests/fixtures/vision/golden_image.png`
//! (256x256, grid 16x16 = 256 patches) — no new fixture asset needed.
//!
//! Every GEMM shape here (patch embed, fused QKV, both attention matmuls,
//! output projection, MLP fc1/fc2) clears the Metal GEMM dispatch threshold
//! (`64*64*64`) at this geometry, so a genuine GPU dispatch happens per op
//! when Metal is available — this is not exercising the CPU-fallback branch.
//!
//! Acquires the machine-wide `gpu_test_lock()` (own copy — external
//! integration tests can't reach the `metal_qwen35.rs`/`metal.rs`-private
//! copies; duplicating this exact lock is the codebase's established
//! sibling-invocation-path pattern for GPU-touching test binaries, see
//! `bin/bench_gdn_prefill_ab.rs`) before any Metal work, and runs
//! `--test-threads=1`-safe (each test acquires its own lock instance).
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16,metal-gpu \
//!     --test vision_s3b_vit_metal_gate_test -- --nocapture --test-threads=1
//! ```

// `fixture_dir`/`cosine_similarity`/`s3b_cfg` live inside `mod gated` (not at
// this top level) because they're only used there — a plain `cargo clippy
// --workspace` build (no `metal-gpu` feature) would otherwise flag them as
// dead code, since the whole `mod gated` block compiles away.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod gated {
    use lattice_inference::model::qwen35_config::VisionModelConfig;
    use lattice_inference::vision::checkpoint::{
        Qwen35VisionWeights, VisualBlockWeights, VisualMergerWeights,
    };
    use lattice_inference::vision::qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward};
    use lattice_inference::vision::qwen35_vit_metal::qwen35_vit_forward_metal;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("vision")
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        let norm_a: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (norm_a * norm_b)
    }

    /// Real per-block ADR-069 `vision_config` geometry, depth reduced to 2 for
    /// test speed (see module docs).
    fn s3b_cfg() -> VisionModelConfig {
        VisionModelConfig {
            depth: 2,
            hidden_size: 768,
            num_heads: 12,
            patch_size: 16,
            spatial_merge_size: 2,
            out_hidden_size: 1024,
            temporal_patch_size: 2,
            num_position_embeddings: 2304,
            in_channels: 3,
            deepstack_visual_indexes: vec![],
        }
    }

    /// Serializes GPU-heavy tests onto the single shared Metal device, both
    /// in-process and machine-wide. Own copy of `metal_qwen35.rs`'s
    /// `gpu_test_lock()` — see module docs for why external integration
    /// tests can't import the original (private, nested in that file's own
    /// `#[cfg(test)] mod tests`).
    struct GpuTestGuard {
        _process: std::sync::MutexGuard<'static, ()>,
        _machine: std::fs::File,
    }

    const GPU_MACHINE_LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
    const GPU_MACHINE_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

    fn gpu_test_lock() -> GpuTestGuard {
        use std::sync::Mutex;
        static GPU_LOCK: Mutex<()> = Mutex::new(());
        let process = GPU_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(GPU_MACHINE_LOCK_PATH)
            .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {GPU_MACHINE_LOCK_PATH}: {e}"));
        let deadline = std::time::Instant::now() + GPU_MACHINE_LOCK_TIMEOUT;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        panic!(
                            "gpu_test_lock: another process has held {GPU_MACHINE_LOCK_PATH} for \
                             over {}s — a Metal test run elsewhere on this machine is wedged or \
                             genuinely that long; inspect `lsof {GPU_MACHINE_LOCK_PATH}`",
                            GPU_MACHINE_LOCK_TIMEOUT.as_secs()
                        );
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(std::fs::TryLockError::Error(e)) => {
                    panic!("gpu_test_lock: flock on {GPU_MACHINE_LOCK_PATH} failed: {e}")
                }
            }
        }

        GpuTestGuard {
            _process: process,
            _machine: file,
        }
    }

    use std::io::Read as _;

    fn make_weights(cfg: &VisionModelConfig, seed: u32) -> Qwen35VisionWeights {
        let hidden = cfg.hidden_size;
        let patch_len = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let mlp_dim = 4 * hidden;
        let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;

        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 0.1 - 0.05
        };
        let mut v = |n: usize| (0..n).map(|_| next()).collect::<Vec<f32>>();

        let make_block = |v: &mut dyn FnMut(usize) -> Vec<f32>| VisualBlockWeights {
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

        let blocks = (0..cfg.depth).map(|_| make_block(&mut v)).collect();

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
            blocks,
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

    fn load_golden_image() -> Vec<u8> {
        let path = fixture_dir().join("golden_image.png");
        let mut f = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).expect("read golden_image.png");
        buf
    }

    /// The S3b gate: Metal forward vs S3a CPU reference forward, same
    /// synthetic weights + committed golden image, cosine > 0.999.
    #[test]
    fn metal_vit_forward_matches_cpu_reference_cosine_gt_0999() {
        let _gpu = gpu_test_lock();

        assert!(
            lattice_inference::forward::metal_gemm::is_available(),
            "Metal device unavailable — S3b gate requires a real Metal device"
        );

        let cfg = s3b_cfg();
        let weights = make_weights(&cfg, 0x1234_5678);
        let image_bytes = load_golden_image();
        let (pixel_values, grid) =
            preprocess_qwen35_image(&image_bytes, &cfg).expect("preprocess golden image");
        assert_eq!(grid.num_patches(), 256);

        let cpu_out =
            qwen35_vit_forward(&weights, &cfg, &pixel_values, grid).expect("cpu reference forward");
        let metal_out =
            qwen35_vit_forward_metal(&weights, &cfg, &pixel_values, grid).expect("metal forward");

        assert_eq!(cpu_out.len(), metal_out.len());
        assert!(metal_out.iter().all(|v| v.is_finite()));

        let cos = cosine_similarity(&cpu_out, &metal_out);
        eprintln!("LATTICE_VISION_S3B_COSINE cosine={cos:.8}");
        assert!(
            cos > 0.999,
            "ADR-069 S3b gate failed: Metal cosine similarity {cos} vs CPU reference must exceed \
             0.999"
        );

        // Mutation-sensitivity proof (CLAUDE.md "Regression Tests Must Be
        // Mutation-Sensitive"): perturbing a single Metal-path-visible weight
        // must drop the gate below threshold.
        let mut mutated = weights.clone();
        for w in mutated.blocks[0].qkv_weight.iter_mut() {
            *w *= -1.0;
        }
        let mutated_metal_out = qwen35_vit_forward_metal(&mutated, &cfg, &pixel_values, grid)
            .expect("mutated metal forward");
        let mutated_cos = cosine_similarity(&cpu_out, &mutated_metal_out);
        eprintln!("LATTICE_VISION_S3B_MUTATION_COSINE cosine={mutated_cos:.8}");
        assert!(
            mutated_cos < 0.999,
            "mutation-sensitivity check failed: negating block[0].qkv_weight did not fail the \
             cosine>0.999 gate (baseline={cos}, mutated={mutated_cos})"
        );
    }
}
