//! `lm_head` Criterion bench group (issue #547).
//!
//! Measures the real final-head kernels the decode path dispatches:
//! - quant path: Q8 tied-f16 wide GEMV vs Q4 GEMV
//! - route: full logits, block-argmax, block-top-k for K in {8, 16, 40, 64}
//!
//! Real measurement only compiles/runs with `--features metal-gpu,f16,bench-internals`
//! on macOS. Default features compile a clean skip so CI (no checkpoint, no GPU
//! feature) stays green:
//!
//! ```bash
//! cargo bench -p lattice-inference --bench lm_head_bench -- "lm_head"
//! cargo bench -p lattice-inference --features metal-gpu,f16,bench-internals --bench lm_head_bench -- "lm_head"
//! ```
//!
//! Checkpoint discovery reuses existing env conventions (no new env vars):
//! `LATTICE_INFERENCE_MODEL_DIR` / `LATTICE_MODEL_DIR` / `LATTICE_TOKENIZER_DIR` /
//! `LATTICE_MODEL_CACHE`, falling back to `~/.lattice/models/...`. Either quant
//! path skips independently (with a logged reason) when its checkpoint is absent.

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "bench-internals"
))]
mod real {
    use criterion::{BenchmarkId, Criterion, black_box};
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::forward::metal_qwen35::bench_support::{
        LmHeadBenchFixture, LmHeadBenchRoute,
    };
    use lattice_inference::model::qwen35::Qwen35Model;
    use lattice_inference::model::qwen35_config::Qwen35Config;
    use std::path::PathBuf;
    use std::time::Duration;

    const MAX_CACHE_LEN: usize = 128;

    #[derive(Clone, Copy, Debug)]
    pub(super) enum QuantPath {
        Q8TiedF16WideGemv,
        Q4Gemv,
    }

    impl QuantPath {
        fn id(self) -> &'static str {
            match self {
                Self::Q8TiedF16WideGemv => "q8_tied_f16_wide_gemv",
                Self::Q4Gemv => "q4_gemv",
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct RouteCase {
        id: &'static str,
        route: LmHeadBenchRoute,
    }

    const ROUTES: [RouteCase; 6] = [
        RouteCase {
            id: "full",
            route: LmHeadBenchRoute::Full,
        },
        RouteCase {
            id: "block_argmax",
            route: LmHeadBenchRoute::BlockArgmax,
        },
        RouteCase {
            id: "block_topk_k8",
            route: LmHeadBenchRoute::BlockTopK { local_k: 8 },
        },
        RouteCase {
            id: "block_topk_k16",
            route: LmHeadBenchRoute::BlockTopK { local_k: 16 },
        },
        RouteCase {
            id: "block_topk_k40",
            route: LmHeadBenchRoute::BlockTopK { local_k: 40 },
        },
        RouteCase {
            id: "block_topk_k64",
            route: LmHeadBenchRoute::BlockTopK { local_k: 64 },
        },
    ];

    fn cache_root() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("LATTICE_MODEL_CACHE") {
            return Some(PathBuf::from(path));
        }
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".lattice/models"))
    }

    fn q8_candidates() -> Vec<PathBuf> {
        if let Some(path) = std::env::var_os("LATTICE_INFERENCE_MODEL_DIR") {
            return vec![PathBuf::from(path)];
        }
        cache_root()
            .map(|root| vec![root.join("qwen3.5-0.8b")])
            .unwrap_or_default()
    }

    fn q4_candidates() -> Vec<PathBuf> {
        if let Some(path) = std::env::var_os("LATTICE_MODEL_DIR") {
            return vec![PathBuf::from(path)];
        }
        cache_root()
            .map(|root| {
                vec![
                    root.join("qwen3.5-0.8b-q4-quarot"),
                    root.join("qwen3.5-0.8b-q4"),
                ]
            })
            .unwrap_or_default()
    }

    fn has_q8_checkpoint(dir: &std::path::Path) -> bool {
        dir.join("config.json").exists()
            && dir.join("tokenizer.json").exists()
            && (dir.join("model.safetensors").exists()
                || dir.join("model.safetensors.index.json").exists())
    }

    fn has_q4_checkpoint(dir: &std::path::Path) -> bool {
        dir.join("config.json").exists()
            && dir
                .join("model_language_model_embed_tokens_weight.q4")
                .exists()
    }

    fn q4_tokenizer_path() -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os("LATTICE_TOKENIZER_DIR") {
            let path = PathBuf::from(dir).join("tokenizer.json");
            return path.exists().then_some(path);
        }
        q8_candidates()
            .into_iter()
            .map(|dir| dir.join("tokenizer.json"))
            .find(|path| path.exists())
    }

    fn display_paths(paths: &[PathBuf]) -> String {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// `Ok(None)` is a clean skip (no checkpoint found — expected in CI).
    /// `Err` means a checkpoint was found but failed to load, which is a
    /// real setup failure the caller should surface, not silently skip.
    pub(super) fn setup_lm_head_fixture(
        quant: QuantPath,
    ) -> Result<Option<LmHeadBenchFixture>, String> {
        match quant {
            QuantPath::Q8TiedF16WideGemv => {
                let candidates = q8_candidates();
                let Some(dir) = candidates.iter().find(|dir| has_q8_checkpoint(dir)) else {
                    eprintln!(
                        "SKIP: lm_head/{} checkpoint not found; searched: {}",
                        quant.id(),
                        display_paths(&candidates)
                    );
                    return Ok(None);
                };
                let model = Qwen35Model::from_safetensors(dir)
                    .map_err(|err| format!("load Q8 safetensors from {}: {err}", dir.display()))?;
                let state = MetalQwen35State::new(model.weights(), model.config(), MAX_CACHE_LEN)
                    .map_err(|err| {
                    format!("create Q8 Metal state from {}: {err}", dir.display())
                })?;
                let mut fixture = LmHeadBenchFixture::new(state);
                fixture.prepare_hidden_for_bench(42, 0);
                Ok(Some(fixture))
            }
            QuantPath::Q4Gemv => {
                let candidates = q4_candidates();
                let Some(dir) = candidates.iter().find(|dir| has_q4_checkpoint(dir)) else {
                    eprintln!(
                        "SKIP: lm_head/{} checkpoint not found; searched: {}",
                        quant.id(),
                        display_paths(&candidates)
                    );
                    return Ok(None);
                };
                let Some(tokenizer_path) = q4_tokenizer_path() else {
                    eprintln!(
                        "SKIP: lm_head/{} tokenizer not found; set LATTICE_TOKENIZER_DIR or keep \
                         tokenizer.json in the Q8 source dir",
                        quant.id()
                    );
                    return Ok(None);
                };
                let cfg = Qwen35Config::from_config_json(&dir.join("config.json"))
                    .map_err(|err| format!("parse Q4 config from {}: {err}", dir.display()))?;
                let state =
                    MetalQwen35State::from_q4_dir(dir, &tokenizer_path, &cfg, MAX_CACHE_LEN)
                        .map_err(|err| {
                            format!(
                                "create Q4 Metal state from {} with tokenizer {}: {err}",
                                dir.display(),
                                tokenizer_path.display()
                            )
                        })?;
                let mut fixture = LmHeadBenchFixture::new(state);
                fixture.prepare_hidden_for_bench(42, 0);
                Ok(Some(fixture))
            }
        }
    }

    pub(super) fn bench_lm_head(c: &mut Criterion) {
        let mut group = c.benchmark_group("lm_head");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));

        for quant in [QuantPath::Q8TiedF16WideGemv, QuantPath::Q4Gemv] {
            let Some(mut fixture) = setup_lm_head_fixture(quant)
                .expect("lm_head benchmark setup failed after checkpoint discovery")
            else {
                continue;
            };

            for case in ROUTES {
                group.bench_function(BenchmarkId::new(quant.id(), case.id), |b| {
                    b.iter(|| {
                        let out = fixture.run_once(case.route);
                        black_box(out.marker);
                    });
                });
            }
        }

        group.finish();
    }
}

#[cfg(not(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "bench-internals"
)))]
fn bench_lm_head(c: &mut Criterion) {
    let _ = c;
    eprintln!(
        "SKIP: lm_head benchmark requires macOS + features metal-gpu,f16,bench-internals \
         (run: cargo bench -p lattice-inference --features metal-gpu,f16,bench-internals \
         --bench lm_head_bench -- \"lm_head\")"
    );
}

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "bench-internals"
))]
fn bench_lm_head(c: &mut Criterion) {
    real::bench_lm_head(c);
}

criterion_group!(benches, bench_lm_head);
criterion_main!(benches);
