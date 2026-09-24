/// Paired RSS probe for lattice#1584 / lattice#1630: does a production Metal dispatch
/// path accumulate autoreleased command buffers when no autorelease pool encloses it?
///
/// Three independent paths, each with its own `nopool`/`pool` pair, selected by
/// `LATTICE_RSS_ARM`. Every arm runs as its own process so neither run inherits another
/// arm's high-water mark:
///
///   decode path (lattice#1584, `forward::metal_qwen35::MetalQwen35State::forward_step`):
///     nopool | pool
///   GEMM path (lattice#1630, `forward::metal_gemm::metal_matmul`, reached from the Metal
///   ViT forward and the serving worker's image-embedding path):
///     gemm-nopool | gemm-pool
///   Qwen3-Embedding path (lattice#1630, `forward::metal::MetalForwardPass::forward`):
///     embed-nopool | embed-pool
///
/// `nopool` calls the path exactly as the library calls it internally today. `pool` adds
/// an extra pool around each call, at the same granularity a per-request pool would give
/// a caller that cannot rely on the library's own internal pooling. Because the library
/// itself now wraps each of these three dispatches in its own per-call autorelease pool
/// (lattice#1584 fixed the decode path; lattice#1630 fixes the other two in the same
/// change this probe ships with), `nopool` is not testing an unpooled binary anymore — it
/// is the regression arm: if a future change ever drops the internal pool, `nopool`'s
/// slope on its own diverges from `pool`'s and this probe is what would catch it. The
/// pair is kept for the same reason lattice#1584's decode pair was kept after its own fix
/// landed: a pool-vs-no-pool comparison is only informative before the fix exists, but
/// removing it after also removes the instrument that would catch a regression.
///
/// Usage (decode path, unchanged from lattice#1584):
///   LATTICE_RSS_ARM=nopool LATTICE_MODEL_DIR="$HOME/.lattice/models/qwen3.5-0.8b" \
///     cargo run --release --example rss_autorelease_probe -p lattice-inference \
///     --features "f16,metal-gpu"
///
/// Usage (GEMM path — no model checkpoint needed, synthetic tensors only):
///   LATTICE_RSS_ARM=gemm-nopool \
///     cargo run --release --example rss_autorelease_probe -p lattice-inference \
///     --features "f16,metal-gpu"
///
/// Usage (Qwen3-Embedding path — synthetic weights at the real
/// `QwenConfig::qwen3_embedding_0_6b()` shape; no checkpoint needed, but the synthesized
/// weight buffers alone are roughly 1.8 GB resident, doubled transiently while the host
/// copy and the uploaded Metal buffer are both live during construction — run this arm on
/// a machine with headroom, and consider a lower `LATTICE_RSS_STEPS` than the default,
/// since a full 28-layer forward pass is much heavier per call than one decode step):
///   LATTICE_RSS_ARM=embed-nopool LATTICE_RSS_STEPS=500 \
///     cargo run --release --example rss_autorelease_probe -p lattice-inference \
///     --features "f16,metal-gpu"
///
/// Env (shared across all arms): LATTICE_RSS_STEPS (default 3000) ·
/// LATTICE_RSS_SAMPLE_EVERY (default 100)
fn main() {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        eprintln!("rss_autorelease_probe requires macOS + metal-gpu.");
        std::process::exit(1);
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    run();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn rss_bytes() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps failed");
    let s = String::from_utf8_lossy(&out.stdout);
    let kib: u64 = s
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("unparseable ps rss {:?}: {e}", s.trim()));
    kib * 1024
}

/// Shared step/sample-count argument parsing, common to every arm.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn rss_probe_step_config() -> (usize, usize) {
    let steps: usize = std::env::var("LATTICE_RSS_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let sample_every: usize = std::env::var("LATTICE_RSS_SAMPLE_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    (steps, sample_every)
}

/// Shared summary line, printed identically by every arm so a downstream reader parses
/// one shape regardless of which path produced it.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn print_rss_summary(
    arm: &str,
    steps: usize,
    samples: &[(usize, u64)],
    elapsed_secs: f64,
    unit: &str,
) {
    let (first_step, first_rss) = samples[0];
    let (last_step, last_rss) = *samples.last().expect("samples");
    let span = (last_step - first_step) as f64;
    let slope = if span > 0.0 {
        (last_rss as f64 - first_rss as f64) / span
    } else {
        0.0
    };
    println!(
        "RSS_SUMMARY arm={arm} steps={steps} samples={} first_rss_bytes={first_rss} \
         last_rss_bytes={last_rss} delta_bytes={} bytes_per_step={slope:.1} \
         {unit}_per_s={:.1}",
        samples.len(),
        last_rss as i64 - first_rss as i64,
        steps as f64 / elapsed_secs
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run() {
    let arm = std::env::var("LATTICE_RSS_ARM").unwrap_or_else(|_| "nopool".to_string());
    match arm.as_str() {
        "nopool" | "pool" => run_decode_arm(&arm),
        "gemm-nopool" | "gemm-pool" => run_gemm_arm(&arm),
        "embed-nopool" | "embed-pool" => run_embed_arm(&arm),
        other => panic!(
            "LATTICE_RSS_ARM must be one of nopool, pool, gemm-nopool, gemm-pool, \
             embed-nopool, embed-pool; got {other:?}"
        ),
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_decode_arm(arm: &str) {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::time::Instant;

    let (steps, sample_every) = rss_probe_step_config();

    let home = std::env::var("HOME").expect("HOME");
    let model_dir_str = std::env::var("LATTICE_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let dir = std::path::Path::new(&model_dir_str);

    let rss_pre_load = rss_bytes();

    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let is_q4 = dir
        .read_dir()
        .map(|mut e| {
            e.any(|x| {
                x.map(|x| x.path().extension().map(|q| q == "q4").unwrap_or(false))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    let prefill_len = 256usize;
    let cache_len = prefill_len + steps + 64;

    let t_load = Instant::now();
    let mut state: MetalQwen35State = if is_q4 {
        let cfg = Qwen35Config::from_model_dir(dir).expect("config.json");
        let tok = dir.join("tokenizer.json");
        MetalQwen35State::from_q4_dir(dir, &tok, &cfg, cache_len).expect("from_q4_dir")
    } else {
        let model = Qwen35Model::from_safetensors(dir).expect("from_safetensors");
        MetalQwen35State::new(model.weights(), model.config(), cache_len).expect("new")
    };
    let rss_post_load = rss_bytes();
    eprintln!(
        "[rss_probe] arm={arm} q4={is_q4} loaded in {:.1}s cache_len={cache_len}",
        t_load.elapsed().as_secs_f64()
    );
    // Positive control for the sampler: loading weights must move RSS by model scale.
    println!(
        "RSS_CONTROL arm={arm} pre_load_bytes={rss_pre_load} post_load_bytes={rss_post_load} \
         delta_bytes={}",
        rss_post_load as i64 - rss_pre_load as i64
    );

    state.reset_state();
    let prompt_ids: Vec<u32> = (0u32..prefill_len as u32)
        .map(|i| 100 + (i % 256))
        .collect();
    let _ = state.forward_prefill(&prompt_ids);

    // Warmup: first steps allocate scratch and caches that are not the subject.
    for i in 0..16usize {
        let pos = prefill_len + i;
        if arm == "pool" {
            objc::rc::autoreleasepool(|| {
                let _ = state.forward_step(200 + (i as u32 % 900), pos);
            });
        } else {
            let _ = state.forward_step(200 + (i as u32 % 900), pos);
        }
    }

    let base_pos = prefill_len + 16;
    let mut samples: Vec<(usize, u64)> = Vec::new();
    let rss_start = rss_bytes();
    samples.push((0, rss_start));
    println!("RSS_SAMPLE arm={arm} step=0 rss_bytes={rss_start}");

    let t0 = Instant::now();
    for i in 0..steps {
        let pos = base_pos + i;
        let tok = 200 + (i as u32 % 900);
        if arm == "pool" {
            objc::rc::autoreleasepool(|| {
                let _ = state.forward_step(tok, pos);
            });
        } else {
            let _ = state.forward_step(tok, pos);
        }
        if (i + 1) % sample_every == 0 {
            let r = rss_bytes();
            samples.push((i + 1, r));
            println!("RSS_SAMPLE arm={arm} step={} rss_bytes={r}", i + 1);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    print_rss_summary(arm, steps, &samples, elapsed, "tok");
}

/// GEMM path (lattice#1630): drives `forward::metal_gemm::run_gemm`'s only production
/// entry point, `metal_matmul`, in a loop. No model checkpoint is needed — `metal_matmul`
/// takes raw `&[f32]` slices, so the shape below is synthesized directly.
///
/// `m=1024, k=n=1152` mirrors the official Qwen3.5-VL vision encoder's hidden size (1152,
/// `crates/inference/src/vision/checkpoint.rs`) at a patch count representative of one
/// moderate-resolution image, well above the GPU dispatch threshold (`64*64*64`) so every
/// call actually reaches `run_gemm` instead of falling back to the CPU path. This is a
/// representative GEMM shape reached from the same call site the issue named
/// (`vision/qwen35_vit_metal.rs`'s per-layer projections via `serve/metal_worker.rs` and
/// `vision/pooled_embed.rs`), not a full ViT forward pass — the autorelease-pool contract
/// `run_gemm` is under test for does not depend on which layer or how many patches drove
/// the call, only on how many times `new_command_buffer` is invoked.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_gemm_arm(arm: &str) {
    use lattice_inference::forward::metal_gemm::metal_matmul;
    use std::time::Instant;

    let (steps, sample_every) = rss_probe_step_config();
    let pooled = arm == "gemm-pool";

    let m = 1024usize;
    let k = 1152usize;
    let n = 1152usize;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 997) as f32) * 1e-3).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 991) as f32) * 1e-3).collect();
    let mut c = vec![0.0f32; m * n];

    let rss_pre = rss_bytes();
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    // Warmup: the first call lazily builds and compiles the Metal pipeline state
    // (`OnceLock` in `metal_gemm::gpu`), which is not the subject of this probe.
    for _ in 0..8usize {
        if pooled {
            objc::rc::autoreleasepool(|| {
                let _ = metal_matmul(&a, &b, &mut c, m, k, n);
            });
        } else {
            let _ = metal_matmul(&a, &b, &mut c, m, k, n);
        }
    }

    let mut samples: Vec<(usize, u64)> = Vec::new();
    let rss_start = rss_bytes();
    samples.push((0, rss_start));
    println!("RSS_SAMPLE arm={arm} step=0 rss_bytes={rss_start}");
    // Positive control for the sampler: the a/b/c allocations plus pipeline compilation
    // must move RSS by more than measurement noise before the timed loop starts.
    println!(
        "RSS_CONTROL arm={arm} pre_setup_bytes={rss_pre} post_warmup_bytes={rss_start} \
         delta_bytes={}",
        rss_start as i64 - rss_pre as i64
    );

    let t0 = Instant::now();
    for i in 0..steps {
        if pooled {
            objc::rc::autoreleasepool(|| {
                let _ = metal_matmul(&a, &b, &mut c, m, k, n);
            });
        } else {
            let _ = metal_matmul(&a, &b, &mut c, m, k, n);
        }
        if (i + 1) % sample_every == 0 {
            let r = rss_bytes();
            samples.push((i + 1, r));
            println!("RSS_SAMPLE arm={arm} step={} rss_bytes={r}", i + 1);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    print_rss_summary(arm, steps, &samples, elapsed, "dispatch");
}

/// Qwen3-Embedding path (lattice#1630): drives `forward::metal::MetalForwardPass::forward`
/// in a loop, at the real `QwenConfig::qwen3_embedding_0_6b()` shape (28 layers,
/// hidden_size=1024, head_dim=128 — the fused-attention kernel accepts no other head_dim).
///
/// No checkpoint is loaded: `MetalForwardPass::new` never reads `QwenWeights::embed_tokens`
/// (the CPU embedding gather happens upstream of this struct, in the caller — see
/// `forward`'s own doc comment), so it is left a zero-length placeholder here to avoid an
/// unneeded ~600 MB allocation for `vocab_size=151669` rows nothing in this path touches.
/// Every weight `forward` itself dispatches against (the 28 per-layer projection and norm
/// matrices) is filled with deterministic non-zero data at its real production shape, the
/// same synthesis approach `metal.rs`'s own
/// `run_fused_attention_with_corrupted_q_lane` test fixture uses at test scale.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_embed_arm(arm: &str) {
    use lattice_inference::forward::metal::MetalForwardPass;
    use lattice_inference::model::qwen::QwenConfig;
    use lattice_inference::weights::{QwenLayerWeights, QwenWeights, Tensor1D, Tensor2D};
    use std::time::Instant;

    let (steps, sample_every) = rss_probe_step_config();
    let pooled = arm == "embed-pool";
    let config = QwenConfig::qwen3_embedding_0_6b();

    fn filled(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.1
            })
            .collect()
    }

    let hidden = config.hidden_size;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let inter = config.intermediate_size;

    #[allow(clippy::type_complexity)]
    let layers_flat: Vec<(
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> = (0..config.num_hidden_layers)
        .map(|i| {
            let seed = 100 + i as u64 * 11;
            (
                filled(q_dim * hidden, seed + 1),
                filled(kv_dim * hidden, seed + 2),
                filled(kv_dim * hidden, seed + 3),
                filled(hidden * q_dim, seed + 4),
                vec![1.0f32; config.head_dim],
                vec![1.0f32; config.head_dim],
                vec![1.0f32; hidden],
                filled(inter * hidden, seed + 5),
                filled(inter * hidden, seed + 6),
                filled(hidden * inter, seed + 7),
                vec![1.0f32; hidden],
            )
        })
        .collect();

    let layers: Vec<QwenLayerWeights<'_>> = layers_flat
        .iter()
        .map(|l| QwenLayerWeights {
            q_proj_weight: Tensor2D {
                data: &l.0,
                rows: q_dim,
                cols: hidden,
            },
            k_proj_weight: Tensor2D {
                data: &l.1,
                rows: kv_dim,
                cols: hidden,
            },
            v_proj_weight: Tensor2D {
                data: &l.2,
                rows: kv_dim,
                cols: hidden,
            },
            o_proj_weight: Tensor2D {
                data: &l.3,
                rows: hidden,
                cols: q_dim,
            },
            q_norm_weight: Tensor1D {
                data: &l.4,
                len: config.head_dim,
            },
            k_norm_weight: Tensor1D {
                data: &l.5,
                len: config.head_dim,
            },
            input_layernorm_weight: Tensor1D {
                data: &l.6,
                len: hidden,
            },
            gate_proj_weight: Tensor2D {
                data: &l.7,
                rows: inter,
                cols: hidden,
            },
            up_proj_weight: Tensor2D {
                data: &l.8,
                rows: inter,
                cols: hidden,
            },
            down_proj_weight: Tensor2D {
                data: &l.9,
                rows: hidden,
                cols: inter,
            },
            post_attention_layernorm_weight: Tensor1D {
                data: &l.10,
                len: hidden,
            },
            fused_qkv: Vec::new(),
            qkv_out_dim: 0,
            fused_gate_up: Vec::new(),
            gate_up_out_dim: 0,
        })
        .collect();

    let norm_weight_flat = vec![1.0f32; hidden];
    // Never read by `MetalForwardPass::new`/`forward` (see the doc comment above) — kept
    // empty rather than allocated at the real ~600 MB vocab_size shape.
    let embed_tokens_flat: Vec<f32> = Vec::new();

    let weights = QwenWeights {
        embed_tokens: Tensor2D {
            data: &embed_tokens_flat,
            rows: 0,
            cols: hidden,
        },
        norm_weight: Tensor1D {
            data: &norm_weight_flat,
            len: hidden,
        },
        layers,
    };

    let rss_pre_load = rss_bytes();
    let _gpu_lock = lattice_inference::measurement::gpu_test_lock();

    let max_seq_len = 64usize;
    let t_load = Instant::now();
    let mut pass =
        MetalForwardPass::new(&config, &weights, max_seq_len).expect("MetalForwardPass::new");
    let rss_post_load = rss_bytes();
    eprintln!(
        "[rss_probe] arm={arm} loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    // Positive control for the sampler: uploading 28 layers of weights to Metal buffers
    // must move RSS by model scale before the timed loop starts.
    println!(
        "RSS_CONTROL arm={arm} pre_load_bytes={rss_pre_load} post_load_bytes={rss_post_load} \
         delta_bytes={}",
        rss_post_load as i64 - rss_pre_load as i64
    );

    // The host-side weight vectors are fully copied into Metal buffers by `new()` above
    // (`make_buffer`/`make_zero_buffer`); nothing below needs them, so they are dropped
    // now rather than held for the rest of this ~1.8 GB-resident process.
    drop(weights);
    drop(layers_flat);
    drop(norm_weight_flat);
    drop(embed_tokens_flat);

    let seq_len = 32usize;
    let hidden_input = filled(seq_len * hidden, 999);

    // Warmup: not the subject, but keeps the timed loop free of first-call effects.
    for _ in 0..4usize {
        if pooled {
            objc::rc::autoreleasepool(|| {
                let _ = pass.forward(&hidden_input, seq_len);
            });
        } else {
            let _ = pass.forward(&hidden_input, seq_len);
        }
    }

    let mut samples: Vec<(usize, u64)> = Vec::new();
    let rss_start = rss_bytes();
    samples.push((0, rss_start));
    println!("RSS_SAMPLE arm={arm} step=0 rss_bytes={rss_start}");

    let t0 = Instant::now();
    for i in 0..steps {
        if pooled {
            objc::rc::autoreleasepool(|| {
                let _ = pass.forward(&hidden_input, seq_len);
            });
        } else {
            let _ = pass.forward(&hidden_input, seq_len);
        }
        if (i + 1) % sample_every == 0 {
            let r = rss_bytes();
            samples.push((i + 1, r));
            println!("RSS_SAMPLE arm={arm} step={} rss_bytes={r}", i + 1);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    print_rss_summary(arm, steps, &samples, elapsed, "forward");
}
