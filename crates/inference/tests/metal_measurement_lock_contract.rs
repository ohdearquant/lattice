use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Attribute, ItemFn, Meta, Token};

#[derive(Clone, Copy)]
enum CallSelector {
    Path(&'static [&'static str]),
    Method(&'static str),
}

const DEVICE_SYSTEM_DEFAULT: CallSelector = CallSelector::Path(&["Device", "system_default"]);
const NEW_COMMAND_BUFFER: CallSelector = CallSelector::Method("new_command_buffer");
const NEW_COMMAND_BUFFER_UNRETAINED: CallSelector =
    CallSelector::Method("new_command_buffer_with_unretained_references");
const SHARED_LOCK_SELECTOR: CallSelector =
    CallSelector::Path(&["lattice_inference", "measurement", "gpu_test_lock"]);
const LOCAL_LOCK_SELECTOR: CallSelector = CallSelector::Path(&["gpu_test_lock"]);
const COMMAND_BUFFER_SELECTORS: &[CallSelector] =
    &[NEW_COMMAND_BUFFER, NEW_COMMAND_BUFFER_UNRETAINED];
const RAW_GPU_SELECTORS: &[CallSelector] = &[
    DEVICE_SYSTEM_DEFAULT,
    NEW_COMMAND_BUFFER,
    NEW_COMMAND_BUFFER_UNRETAINED,
];
const SHARED_LOCK_CALL: &str = "lattice_inference::measurement::gpu_test_lock()";
const CHECKED_CARGO_TARGET_KINDS: &[&str] = &["bench", "bin", "example", "lib", "test"];
const RAW_HARNESS_ENTRYPOINTS: &[(&str, CallSelector)] = &[
    (
        "benches/decode_attn_bench.rs",
        CallSelector::Path(&["bench_flash_decode"]),
    ),
    (
        "benches/topk_readback.rs",
        CallSelector::Path(&["bench_topk_parity"]),
    ),
    ("examples/bench_concurrent.rs", CallSelector::Path(&["run"])),
    ("examples/bench_dispatch.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_dispatch2.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_mps_gemm.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/bench_simdgroup.rs", DEVICE_SYSTEM_DEFAULT),
    ("examples/profile_metal_decode.rs", DEVICE_SYSTEM_DEFAULT),
];
const ADDITIONAL_GUARDED_ENTRYPOINTS: &[(&str, CallSelector)] = &[
    (
        "examples/bench_embed_quality.rs",
        CallSelector::Path(&["run_bench"]),
    ),
    (
        "examples/bench_metal.rs",
        CallSelector::Path(&["MetalForwardPass", "new"]),
    ),
];

struct BenchHarnessCall {
    selector: CallSelector,
    callers: &'static [&'static str],
    guarded: bool,
}

struct BenchHarness {
    path: &'static str,
    registered: &'static [&'static str],
    calls: &'static [BenchHarnessCall],
    constructions: &'static [(&'static str, CallSelector)],
}

// These are finite, reviewed caller chains, not Rust name resolution. The
// syntax check below rejects aliases, deferred calls, and other spellings of
// these names rather than silently losing the ownership proof.
const BENCH_HARNESSES: &[BenchHarness] = &[
    BenchHarness {
        path: "benches/metal_decode_bench.rs",
        registered: &["bench_locked_metal_decode"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_metal_decode_q4"]),
                callers: &["bench_locked_metal_decode"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_metal_decode_q8"]),
                callers: &["bench_locked_metal_decode"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_metal_prefill_q4"]),
                callers: &["bench_locked_metal_decode"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["load_q4_state"]),
                callers: &["bench_metal_decode_q4", "bench_metal_prefill_q4"],
                guarded: false,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["load_q8_state"]),
                callers: &["bench_metal_decode_q8"],
                guarded: false,
            },
        ],
        constructions: &[
            (
                "load_q4_state",
                CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
            ),
            (
                "load_q8_state",
                CallSelector::Path(&["MetalQwen35State", "new"]),
            ),
        ],
    },
    BenchHarness {
        path: "benches/cross_turn_prefix_cache_bench.rs",
        registered: &["bench_locked_cross_turn_prefix_cache"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_cross_turn_prefix_cache"]),
                callers: &["bench_locked_cross_turn_prefix_cache"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["load_state_and_tokenizer"]),
                callers: &["bench_cross_turn_prefix_cache"],
                guarded: false,
            },
        ],
        constructions: &[(
            "load_state_and_tokenizer",
            CallSelector::Path(&["MetalQwen35State", "new"]),
        )],
    },
    BenchHarness {
        path: "benches/lm_head_bench.rs",
        registered: &["bench_locked_lm_head"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_lm_head"]),
                callers: &["bench_locked_lm_head"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["real", "bench_lm_head"]),
                callers: &["bench_lm_head"],
                guarded: false,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["setup_lm_head_fixture"]),
                callers: &["real::bench_lm_head"],
                guarded: false,
            },
        ],
        constructions: &[
            (
                "real::setup_lm_head_fixture",
                CallSelector::Path(&["MetalQwen35State", "new"]),
            ),
            (
                "real::setup_lm_head_fixture",
                CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
            ),
        ],
    },
    BenchHarness {
        path: "benches/mtp_decode.rs",
        registered: &["bench_locked_mtp_decode"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_baseline"]),
                callers: &["bench_locked_mtp_decode"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_mtp"]),
                callers: &["bench_locked_mtp_decode"],
                guarded: true,
            },
        ],
        constructions: &[
            (
                "bench_baseline",
                CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
            ),
            (
                "bench_mtp",
                CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
            ),
        ],
    },
    BenchHarness {
        path: "benches/decode_attn_bench.rs",
        registered: &["bench_locked_metal_decode"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_reference_decode"]),
                callers: &["bench_locked_metal_decode"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_flash_decode"]),
                callers: &["bench_locked_metal_decode"],
                guarded: true,
            },
        ],
        constructions: &[],
    },
    BenchHarness {
        path: "benches/topk_readback.rs",
        registered: &["bench_locked_topk_readback"],
        calls: &[
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_full_logit_readback"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_compact_readback"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_sampling_pipeline"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_topk_selection"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_full_logit_readback_metal"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_compact_readback_metal"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_noop_command_buffer"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_metal_topk_dispatch_only"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_metal_topk_plus_readback"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_topk_parity"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
            BenchHarnessCall {
                selector: CallSelector::Path(&["bench_candidate_sampler"]),
                callers: &["bench_locked_topk_readback"],
                guarded: true,
            },
        ],
        constructions: &[],
    },
];
const TARGETS_WITH_RECOGNIZED_METAL_MARKERS: &[&str] = &[
    "benches/cross_turn_prefix_cache_bench.rs",
    "benches/decode_attn_bench.rs",
    "benches/lm_head_bench.rs",
    "benches/metal_decode_bench.rs",
    "benches/mtp_decode.rs",
    "benches/topk_readback.rs",
    "examples/bench_concurrent.rs",
    "examples/bench_dispatch.rs",
    "examples/bench_dispatch2.rs",
    "examples/bench_embed_quality.rs",
    "examples/bench_embedding.rs",
    "examples/bench_gdn_decode.rs",
    "examples/bench_gdn_prefill_ab.rs",
    "examples/bench_gdn_state.rs",
    "examples/bench_gpu.rs",
    "examples/bench_metal.rs",
    "examples/bench_mps_gemm.rs",
    "examples/bench_persistent_state.rs",
    "examples/bench_profile.rs",
    "examples/bench_pruning.rs",
    "examples/bench_q4_prefill.rs",
    "examples/bench_q8_prefill.rs",
    "examples/bench_quality.rs",
    "examples/bench_simdgroup.rs",
    "examples/bench_stability.rs",
    "examples/bench_suite.rs",
    "examples/decode_profile.rs",
    "examples/layer_sweep.rs",
    "examples/profile_metal.rs",
    "examples/profile_metal_decode.rs",
    "src/bin/backfill_qwen3.rs",
    "src/bin/bench_decode_ab.rs",
    "src/bin/bench_decode_slopefit.rs",
    "src/bin/bench_gdn_prefill_ab.rs",
    "src/bin/bench_logit_dump.rs",
    "src/bin/bench_lora_mixture.rs",
    "src/bin/bench_vision_prefill_ab.rs",
    "src/bin/chat_metal.rs",
    "src/bin/dump_quarot_q4_golden.rs",
    "src/bin/eval_perplexity.rs",
    "src/bin/gramperf_profile.rs",
    "src/bin/lattice/main.rs",
    "src/bin/lattice_serve.rs",
    "src/bin/ppl_metal.rs",
];
const TARGETS_WITHOUT_RECOGNIZED_METAL_MARKERS: &[&str] = &[
    "benches/attention_dispatch_bench.rs",
    "benches/attn_opt_bench.rs",
    "benches/backward_stage0.rs",
    "benches/differential_attention_bench.rs",
    "benches/e2e_bench.rs",
    "benches/elementwise_bench.rs",
    "benches/elementwise_cpu_bench.rs",
    "benches/f16_convert_bench.rs",
    "benches/gated_attention_bench.rs",
    "benches/grammar_mask_bench.rs",
    "benches/inference_bench.rs",
    "benches/inference_perf.rs",
    "benches/kv_cache_f16_bench.rs",
    "benches/kv_cache_layout_bench.rs",
    "benches/metrics_bench.rs",
    "benches/native_sparse_attention_bench.rs",
    "benches/pruning_bench.rs",
    "benches/quarot_hadamard_bench.rs",
    "benches/tokenizer_bench.rs",
    "examples/bench_gdn.rs",
    "examples/diff_attn_layer23.rs",
    "examples/diff_gdn_layer.rs",
    "examples/ernie45_trace_dump.rs",
    "examples/load_cross_encoder.rs",
    "src/bin/gramtime_profile.rs",
    "src/bin/moe_admission_sim.rs",
    "src/bin/quantize_q4.rs",
    "src/bin/quantize_quarot.rs",
    "src/bin/qwen35_debug.rs",
    "src/bin/qwen35_generate.rs",
];
const IN_CRATE_COMMAND_BUFFER_TESTS: &[&str] = &[
    "src/forward/metal.rs::rms_norm_matches_pre_854_oracle_on_finite_input",
    "src/forward/metal_qwen35.rs::test_gemv_decode_numerical",
    "src/forward/metal_qwen35.rs::test_gpu_argmax_parity_k1",
    "src/forward/metal_qwen35.rs::decode_attention_reference_fails_closed_on_nan_q",
    "src/forward/metal_qwen35.rs::gemv_q3_decode_matches_cpu_reference_across_shapes",
    "src/forward/metal_qwen35.rs::gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries",
    "src/forward/metal_qwen35.rs::gemm_q3_tiled_differential_fails_closed_on_nan_scale",
    "src/forward/metal_qwen35.rs::gemv_q3_decode_mutation_sensitive_high_plane_bit",
    "src/forward/metal_qwen35.rs::lora_gemv_kernel_matches_cpu_reference",
    "src/forward/metal_qwen35.rs::load_adapter_and_dispatch_lora_if_active",
    "src/forward/metal_qwen35.rs::forced_non_apple7_q4_gemm_fallback_dispatches_and_matches_reference",
    "src/forward/metal_qwen35/inner/tests/dispatch.rs::dispatch_matmul_q4_writes_all_rows",
];
const CONSTRUCTION_SELECTORS: &[CallSelector] = &[
    CallSelector::Path(&["MetalQwen35State", "new"]),
    CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
    // QwenModel::from_directory attempts MetalForwardPass::new internally
    // (crates/inference/src/model/qwen.rs) whenever Metal is available, so
    // every caller of it is a helper-mediated Metal construction site (#1274).
    CallSelector::Path(&["QwenModel", "from_directory"]),
];
/// A construction site exempted from the live shared-lock requirement.
///
/// `site` is a stable key built from the enclosing `mod`/`impl`/`fn` path and
/// the constructed selector (see `StructuredSource::stable_construction_key`),
/// not from source line/column, so edits that shift line numbers without
/// changing what is constructed or where it sits in the program structure do
/// not invalidate the entry (#1357). `recorded_position` is a line:column
/// snapshot taken when the entry was authored, kept only as a tripwire: if a
/// site's current position no longer matches it, the site has moved and is
/// worth a human look, even though its stable key still matches.
struct ConstructionExemption {
    site: &'static str,
    recorded_position: &'static str,
    reason: &'static str,
}

const CONSTRUCTION_EXEMPTIONS: &[ConstructionExemption] = &[
    ConstructionExemption {
        site: "example:bench_concurrent:examples/bench_concurrent.rs=>examples/bench_concurrent.rs::run::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "examples/bench_concurrent.rs:444:27",
        reason: "run state1 construction is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        site: "example:bench_concurrent:examples/bench_concurrent.rs=>examples/bench_concurrent.rs::run::MetalQwen35State::from_q4_dir()#2",
        recorded_position: "examples/bench_concurrent.rs:474:27",
        reason: "run state2 construction is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        site: "example:bench_embed_quality:examples/bench_embed_quality.rs=>examples/bench_embed_quality.rs::run_bench::QwenModel::from_directory()#1",
        recorded_position: "examples/bench_embed_quality.rs:136:52",
        reason: "run_bench QwenModel::from_directory is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        site: "bin:backfill_qwen3:src/bin/backfill_qwen3.rs=>src/bin/backfill_qwen3.rs::main::QwenModel::from_directory()#1",
        recorded_position: "src/bin/backfill_qwen3.rs:37:28",
        reason: "one-time DB embedding migration over an unbounded row count belongs to a long-running batch process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:chat_metal:src/bin/chat_metal.rs=>src/bin/chat_metal.rs::run::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/chat_metal.rs:801:39",
        reason: "run Q4 initialization belongs to a long-running interactive process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:chat_metal:src/bin/chat_metal.rs=>src/bin/chat_metal.rs::run::MetalQwen35State::new()#1",
        recorded_position: "src/bin/chat_metal.rs:826:39",
        reason: "run safetensors initialization belongs to a long-running interactive process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:lattice:src/bin/lattice/main.rs=>src/bin/lattice/chat.rs::MetalChatBackend::load::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/lattice/chat.rs:53:81",
        reason: "MetalChatBackend::load belongs to a long-running interactive process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:lattice:src/bin/lattice/main.rs=>src/bin/lattice/serve.rs::ModelBackend::spawn_metal::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/lattice/serve.rs:423:81",
        reason: "ModelBackend::spawn_metal initializes a long-running server worker outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:lattice_serve:src/bin/lattice_serve.rs=>src/bin/lattice_serve.rs::imp::load_model::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/lattice_serve.rs:1795:47",
        reason: "load_model Q4 initialization belongs to a long-running server outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        site: "bin:lattice_serve:src/bin/lattice_serve.rs=>src/bin/lattice_serve.rs::imp::load_model::MetalQwen35State::new()#1",
        recorded_position: "src/bin/lattice_serve.rs:1815:47",
        reason: "load_model safetensors initialization belongs to a long-running server outside the bounded measurement-harness contract",
    },
];

fn rust_sources_under(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(path);
            }
        }
    }
    sources.sort();
    sources
}

#[derive(Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
}

#[derive(Deserialize)]
struct CargoPackage {
    manifest_path: PathBuf,
    targets: Vec<CargoMetadataTarget>,
}

#[derive(Deserialize)]
struct CargoMetadataTarget {
    name: String,
    kind: Vec<String>,
    src_path: PathBuf,
}

struct CargoTargetRoot {
    name: String,
    kind: String,
    path: PathBuf,
}

fn cargo_targets(manifest_dir: &Path, kinds: &[&str]) -> Result<Vec<CargoTargetRoot>, String> {
    let manifest_path = manifest_dir.join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--manifest-path",
        ])
        .arg(&manifest_path)
        .output()
        .map_err(|reason| format!("could not run cargo metadata: {reason}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed for {}: {}",
            manifest_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let metadata: CargoMetadata = serde_json::from_slice(&output.stdout)
        .map_err(|reason| format!("cargo metadata output could not be parsed: {reason}"))?;
    let expected_manifest = std::fs::canonicalize(&manifest_path).map_err(|reason| {
        format!(
            "could not resolve package manifest {}: {reason}",
            manifest_path.display()
        )
    })?;
    let mut matching_packages = Vec::new();
    for package in metadata.packages {
        let package_manifest = std::fs::canonicalize(&package.manifest_path).map_err(|reason| {
            format!(
                "could not resolve cargo metadata manifest {}: {reason}",
                package.manifest_path.display()
            )
        })?;
        if package_manifest == expected_manifest {
            matching_packages.push(package);
        }
    }
    let mut packages = matching_packages.into_iter();
    let package = packages.next().ok_or_else(|| {
        format!(
            "cargo metadata did not contain package manifest {}",
            expected_manifest.display()
        )
    })?;
    if packages.next().is_some() {
        return Err(format!(
            "cargo metadata contained duplicate package manifest {}",
            expected_manifest.display()
        ));
    }

    let requested = kinds.iter().copied().collect::<BTreeSet<_>>();
    let checked_kinds = CHECKED_CARGO_TARGET_KINDS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut targets = Vec::new();
    for target in package.targets {
        if !target
            .kind
            .iter()
            .any(|kind| checked_kinds.contains(kind.as_str()))
        {
            return Err(format!(
                "cargo target {} has no checked target kind: {}",
                target.src_path.display(),
                target.kind.join(", ")
            ));
        }
        let matching = target
            .kind
            .iter()
            .filter(|kind| requested.contains(kind.as_str()))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        if matching.len() != 1 {
            return Err(format!(
                "cargo target {} has ambiguous requested kinds: {}",
                target.src_path.display(),
                matching
                    .iter()
                    .map(|kind| kind.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let path = std::fs::canonicalize(&target.src_path).map_err(|reason| {
            format!(
                "could not resolve Cargo target source {}: {reason}",
                target.src_path.display()
            )
        })?;
        targets.push(CargoTargetRoot {
            name: target.name,
            kind: matching[0].clone(),
            path,
        });
    }
    targets.sort_by(|left, right| left.path.cmp(&right.path));
    if targets.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err("cargo metadata selected the same target source more than once".to_string());
    }
    Ok(targets)
}

fn cargo_target_roots(manifest_dir: &Path, kinds: &[&str]) -> Result<Vec<PathBuf>, String> {
    cargo_targets(manifest_dir, kinds)
        .map(|targets| targets.into_iter().map(|target| target.path).collect())
}

fn assert_construction_inventory_classified(
    discovered: &BTreeSet<String>,
    classified: &BTreeSet<String>,
) {
    assert_eq!(
        discovered, classified,
        "MetalQwen35State construction inventory changed; every site must acquire the shared \
         lock before construction or have an explicit, justified exemption"
    );
}

fn assert_guard_fixture_rejected(source: &str, context: &str) {
    let result = StructuredSource::parse(context, source, false).and_then(|parsed| {
        parsed.validate_selector(
            0..parsed.tokens.len(),
            DEVICE_SYSTEM_DEFAULT,
            SHARED_LOCK_SELECTOR,
            GuardRequirement::Lexical,
        )
    });
    let Err(message) = result else {
        panic!("source contract accepted {context}");
    };
    assert!(
        message.contains(context),
        "source contract failure did not name {context}: {message}"
    );
}

fn assert_guard_fixture_accepted(source: &str, context: &str) {
    let parsed =
        StructuredSource::parse(context, source, false).unwrap_or_else(|reason| panic!("{reason}"));
    parsed
        .validate_selector(
            0..parsed.tokens.len(),
            DEVICE_SYSTEM_DEFAULT,
            SHARED_LOCK_SELECTOR,
            GuardRequirement::Lexical,
        )
        .unwrap_or_else(|reason| panic!("source contract rejected {context}: {reason}"));
}

fn command_buffer_fixture_result(source: &str, context: &str) -> Result<(), String> {
    StructuredSource::parse(context, source, true).and_then(|parsed| {
        let functions = parsed.test_functions()?;
        parsed.reject_selector_in_macros(NEW_COMMAND_BUFFER)?;
        let mut found = false;
        for function in functions {
            let calls = parsed.call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)?;
            if calls.is_empty() {
                continue;
            }
            found = true;
            parsed.validate_work_sites(
                &calls,
                LOCAL_LOCK_SELECTOR,
                GuardRequirement::Function {
                    closing_brace: function.body.end,
                },
            )?;
        }
        if found {
            Ok(())
        } else {
            Err(format!("{context}: protected work was not classified"))
        }
    })
}

fn assert_command_buffer_fixture_rejected_with(
    source: &str,
    context: &str,
    expected_failure: &str,
) {
    let Err(message) = command_buffer_fixture_result(source, context) else {
        panic!("source contract accepted {context}");
    };
    assert!(
        message.contains(expected_failure),
        "source contract reported the wrong failure for {context}; expected `{expected_failure}`, got: {message}"
    );
}

#[derive(Clone, Debug)]
enum RustTokenKind {
    Ident { text: String, raw: bool },
    String(String),
    Punct(char),
    Other,
}

#[derive(Clone, Debug)]
struct RustToken {
    kind: RustTokenKind,
    offset: usize,
}

impl RustToken {
    fn ident(&self) -> Option<&str> {
        match &self.kind {
            RustTokenKind::Ident { text, .. } => Some(text),
            _ => None,
        }
    }

    fn is_raw_ident(&self) -> bool {
        matches!(self.kind, RustTokenKind::Ident { raw: true, .. })
    }

    fn is_punct(&self, expected: char) -> bool {
        matches!(self.kind, RustTokenKind::Punct(actual) if actual == expected)
    }
}

fn raw_string_at(source: &str, start: usize) -> Option<Result<(usize, String), String>> {
    let bytes = source.as_bytes();
    let mut cursor = start;
    if matches!(bytes.get(cursor), Some(b'b' | b'c')) {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    let hashes = cursor - hashes_start;
    let content_start = cursor + 1;
    cursor = content_start;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"'
            && bytes.get(cursor + 1..cursor + 1 + hashes)
                == Some(&bytes[hashes_start..hashes_start + hashes])
        {
            return Some(Ok((
                cursor + 1 + hashes,
                source[content_start..cursor].to_string(),
            )));
        }
        cursor += 1;
    }
    Some(Err(format!("unterminated raw string at byte {start}")))
}

fn quoted_string_at(source: &str, quote: usize) -> Result<(usize, String), String> {
    let bytes = source.as_bytes();
    let mut cursor = quote + 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\\' {
            cursor = (cursor + 2).min(bytes.len());
        } else if bytes[cursor] == b'"' {
            return Ok((cursor + 1, source[quote + 1..cursor].to_string()));
        } else {
            cursor += 1;
        }
    }
    Err(format!("unterminated string at byte {quote}"))
}

fn char_literal_end(source: &str, quote: usize) -> Option<Result<usize, String>> {
    let bytes = source.as_bytes();
    if bytes.get(quote) != Some(&b'\'') {
        return None;
    }
    let mut cursor = quote + 1;
    if bytes.get(cursor) == Some(&b'\\') {
        cursor += 1;
        while cursor < bytes.len() {
            if bytes[cursor] == b'\'' {
                return Some(Ok(cursor + 1));
            }
            cursor += 1;
        }
        return Some(Err(format!(
            "unterminated character literal at byte {quote}"
        )));
    }
    let character = source[cursor..].chars().next()?;
    cursor += character.len_utf8();
    (bytes.get(cursor) == Some(&b'\'')).then_some(Ok(cursor + 1))
}

fn rust_tokens(source: &str) -> Result<Vec<RustToken>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if bytes[offset].is_ascii_whitespace() {
            offset += 1;
            continue;
        }
        if bytes[offset..].starts_with(b"//") {
            offset = bytes[offset..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |length| offset + length + 1);
            continue;
        }
        if bytes[offset..].starts_with(b"/*") {
            let start = offset;
            offset += 2;
            let mut depth = 1usize;
            while offset < bytes.len() && depth > 0 {
                if bytes[offset..].starts_with(b"/*") {
                    depth += 1;
                    offset += 2;
                } else if bytes[offset..].starts_with(b"*/") {
                    depth -= 1;
                    offset += 2;
                } else {
                    offset += 1;
                }
            }
            if depth != 0 {
                return Err(format!("unterminated block comment at byte {start}"));
            }
            continue;
        }
        if let Some(raw) = raw_string_at(source, offset) {
            let (end, value) = raw?;
            tokens.push(RustToken {
                kind: RustTokenKind::String(value),
                offset,
            });
            offset = end;
            continue;
        }
        let quote = if bytes[offset] == b'"' {
            Some(offset)
        } else if matches!(bytes[offset], b'b' | b'c') && bytes.get(offset + 1) == Some(&b'"') {
            Some(offset + 1)
        } else {
            None
        };
        if let Some(quote) = quote {
            let (end, value) = quoted_string_at(source, quote)?;
            tokens.push(RustToken {
                kind: RustTokenKind::String(value),
                offset,
            });
            offset = end;
            continue;
        }
        let char_quote = if bytes[offset] == b'\'' {
            Some(offset)
        } else if bytes[offset] == b'b' && bytes.get(offset + 1) == Some(&b'\'') {
            Some(offset + 1)
        } else {
            None
        };
        if let Some(quote) = char_quote
            && let Some(end) = char_literal_end(source, quote)
        {
            tokens.push(RustToken {
                kind: RustTokenKind::Other,
                offset,
            });
            offset = end?;
            continue;
        }
        if bytes[offset..].starts_with(b"r#")
            && source[offset + 2..]
                .chars()
                .next()
                .is_some_and(|character| character == '_' || character.is_alphabetic())
        {
            let mut end = offset + 2;
            for character in source[end..].chars() {
                if character == '_' || character.is_alphanumeric() {
                    end += character.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(RustToken {
                kind: RustTokenKind::Ident {
                    text: source[offset + 2..end].to_string(),
                    raw: true,
                },
                offset,
            });
            offset = end;
            continue;
        }
        let character = source[offset..]
            .chars()
            .next()
            .ok_or_else(|| format!("invalid UTF-8 boundary at byte {offset}"))?;
        if character == '_' || character.is_alphabetic() {
            let mut end = offset + character.len_utf8();
            for character in source[end..].chars() {
                if character == '_' || character.is_alphanumeric() {
                    end += character.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(RustToken {
                kind: RustTokenKind::Ident {
                    text: source[offset..end].to_string(),
                    raw: false,
                },
                offset,
            });
            offset = end;
            continue;
        }
        tokens.push(RustToken {
            kind: if character.is_ascii_punctuation() {
                RustTokenKind::Punct(character)
            } else {
                RustTokenKind::Other
            },
            offset,
        });
        offset += character.len_utf8();
    }
    Ok(tokens)
}

type DelimiterStructure = (Vec<Option<usize>>, Vec<Option<usize>>);

fn delimiter_structure(tokens: &[RustToken]) -> Result<DelimiterStructure, String> {
    let mut pairs = vec![None; tokens.len()];
    let mut parents = vec![None; tokens.len()];
    let mut stack = Vec::<(char, usize)>::new();
    for (index, token) in tokens.iter().enumerate() {
        match token.kind {
            RustTokenKind::Punct(open @ ('(' | '[' | '{')) => {
                parents[index] = stack.last().map(|(_, parent)| *parent);
                stack.push((open, index));
            }
            RustTokenKind::Punct(close @ (')' | ']' | '}')) => {
                let Some((open, opening)) = stack.pop() else {
                    return Err(format!("unmatched `{close}` at byte {}", token.offset));
                };
                let expected = match open {
                    '(' => ')',
                    '[' => ']',
                    '{' => '}',
                    _ => unreachable!(),
                };
                if close != expected {
                    return Err(format!(
                        "mismatched `{open}` and `{close}` at byte {}",
                        token.offset
                    ));
                }
                pairs[opening] = Some(index);
                pairs[index] = Some(opening);
                parents[index] = parents[opening];
            }
            _ => parents[index] = stack.last().map(|(_, parent)| *parent),
        }
    }
    if let Some((open, opening)) = stack.pop() {
        return Err(format!(
            "unclosed `{open}` at byte {}",
            tokens[opening].offset
        ));
    }
    Ok((pairs, parents))
}

#[derive(Clone, Debug)]
enum CfgFormula {
    True,
    False,
    Atom(String),
    Not(Box<CfgFormula>),
    All(Vec<CfgFormula>),
    Any(Vec<CfgFormula>),
}

impl CfgFormula {
    fn all(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::True => {}
                Self::False => return Self::False,
                Self::All(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::True,
            1 => combined.pop().unwrap_or(Self::True),
            _ => Self::All(combined),
        }
    }

    fn any(formulas: impl IntoIterator<Item = CfgFormula>) -> Self {
        let mut combined = Vec::new();
        for formula in formulas {
            match formula {
                Self::False => {}
                Self::True => return Self::True,
                Self::Any(nested) => combined.extend(nested),
                other => combined.push(other),
            }
        }
        match combined.len() {
            0 => Self::False,
            1 => combined.pop().unwrap_or(Self::False),
            _ => Self::Any(combined),
        }
    }

    fn not(formula: CfgFormula) -> Self {
        match formula {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Not(inner) => *inner,
            other => Self::Not(Box::new(other)),
        }
    }

    fn collect_atoms(&self, atoms: &mut BTreeSet<String>) {
        match self {
            Self::Atom(atom) => {
                atoms.insert(atom.clone());
            }
            Self::Not(formula) => formula.collect_atoms(atoms),
            Self::All(formulas) | Self::Any(formulas) => {
                for formula in formulas {
                    formula.collect_atoms(atoms);
                }
            }
            Self::True | Self::False => {}
        }
    }

    fn evaluate(&self, values: &std::collections::BTreeMap<String, bool>) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::Atom(atom) => values.get(atom).copied().unwrap_or(false),
            Self::Not(formula) => !formula.evaluate(values),
            Self::All(formulas) => formulas.iter().all(|formula| formula.evaluate(values)),
            Self::Any(formulas) => formulas.iter().any(|formula| formula.evaluate(values)),
        }
    }

    fn satisfiable(&self) -> Result<bool, String> {
        let mut atoms = BTreeSet::new();
        self.collect_atoms(&mut atoms);
        if atoms.len() > 16 {
            return Err(format!(
                "cfg expression has {} independent atoms; refusing an unbounded classification",
                atoms.len()
            ));
        }
        let atoms = atoms.into_iter().collect::<Vec<_>>();
        for assignment in 0usize..(1usize << atoms.len()) {
            let values = atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| (atom.clone(), assignment & (1 << index) != 0))
                .collect::<std::collections::BTreeMap<_, _>>();
            if self.evaluate(&values) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn implies(&self, other: &CfgFormula) -> Result<bool, String> {
        Ok(!Self::all([self.clone(), Self::not(other.clone())]).satisfiable()?)
    }
}

#[derive(Clone, Debug)]
struct AttributeSpec {
    content: Range<usize>,
}

#[derive(Clone, Debug)]
struct FunctionSpec {
    name: String,
    body: Range<usize>,
    returns_unit: bool,
    attributes: Vec<AttributeSpec>,
    test_registration: TestRegistration,
    unclassifiable_macro: Option<String>,
}

#[derive(Clone, Debug)]
enum TestRegistration {
    No,
    Yes,
    Unclassifiable(String),
}

#[derive(Clone, Debug)]
struct ScopeSpec {
    body: Range<usize>,
    attributes: Vec<AttributeSpec>,
}

struct StructuredSource {
    context: String,
    source: String,
    tokens: Vec<RustToken>,
    pairs: Vec<Option<usize>>,
    parents: Vec<Option<usize>>,
    macro_ranges: Vec<Range<usize>>,
    functions: Vec<FunctionSpec>,
    /// Token-index body range paired with a `mod::...::Type::method` path,
    /// for every free function and `impl` method in the file (nested modules
    /// and `impl` blocks included). Used to name a construction site by the
    /// program structure enclosing it instead of by source position.
    function_paths: Vec<(Range<usize>, String)>,
    scopes: Vec<ScopeSpec>,
    file_attributes: Vec<AttributeSpec>,
    test_cfg: bool,
    external_cfg: CfgFormula,
    unclassifiable_test_scope: Option<String>,
}

fn path_label(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn classify_test_meta(meta: &Meta) -> TestRegistration {
    let path = meta.path();
    if path.is_ident("test") {
        return TestRegistration::Yes;
    }
    if path_label(path) == "tokio::test" {
        return TestRegistration::Yes;
    }
    if path.is_ident("cfg_attr") {
        let Meta::List(list) = meta else {
            return TestRegistration::Unclassifiable("cfg_attr".to_string());
        };
        let Ok(arguments) =
            Punctuated::<Meta, Token![,]>::parse_terminated.parse2(list.tokens.clone())
        else {
            return TestRegistration::Unclassifiable("cfg_attr".to_string());
        };
        let mut registration = TestRegistration::No;
        for emitted in arguments.iter().skip(1) {
            match classify_test_meta(emitted) {
                TestRegistration::No => {}
                TestRegistration::Yes => registration = TestRegistration::Yes,
                unclassifiable @ TestRegistration::Unclassifiable(_) => {
                    return unclassifiable;
                }
            }
        }
        return registration;
    }

    let label = path_label(path);
    if path.segments.len() == 1
        && matches!(
            label.as_str(),
            "allow"
                | "cfg"
                | "cold"
                | "deprecated"
                | "doc"
                | "forbid"
                | "ignore"
                | "inline"
                | "must_use"
                | "should_panic"
                | "target_feature"
                | "track_caller"
                | "unsafe"
                | "warn"
        )
    {
        TestRegistration::No
    } else {
        TestRegistration::Unclassifiable(label)
    }
}

fn classify_test_attributes(attributes: &[Attribute]) -> TestRegistration {
    let mut registration = TestRegistration::No;
    for attribute in attributes {
        match classify_test_meta(&attribute.meta) {
            TestRegistration::No => {}
            TestRegistration::Yes => registration = TestRegistration::Yes,
            unclassifiable @ TestRegistration::Unclassifiable(_) => return unclassifiable,
        }
    }
    registration
}

fn macro_tokens_name_protected_work(tokens: &proc_macro2::TokenStream) -> bool {
    let raw_dispatch = tokens.clone().into_iter().any(|token| match token {
        proc_macro2::TokenTree::Group(group) => macro_tokens_name_protected_work(&group.stream()),
        proc_macro2::TokenTree::Ident(ident) => matches!(
            ident.to_string().as_str(),
            "new_command_buffer"
                | "new_command_buffer_with_unretained_references"
                | "system_default"
        ),
        _ => false,
    });
    if raw_dispatch {
        return true;
    }
    let rendered = tokens.to_string();
    [
        "MetalForwardPass :: new",
        "MetalQwen35State :: from_q4_dir",
        "MetalQwen35State :: new",
        "QwenModel :: from_directory",
    ]
    .iter()
    .any(|selector| rendered.contains(selector))
}

fn unclassifiable_macro(mac: &syn::Macro) -> Option<String> {
    let label = path_label(&mac.path);
    if label == "include" {
        return Some("unclassifiable include! in test-bearing scope".to_string());
    }
    let name = mac.path.segments.last()?.ident.to_string();
    if matches!(
        name.as_str(),
        "cfg"
            | "column"
            | "concat"
            | "env"
            | "file"
            | "include_bytes"
            | "include_str"
            | "line"
            | "module_path"
            | "option_env"
            | "oslogstring"
            | "stringify"
    ) {
        return None;
    }
    if matches!(
        name.as_str(),
        "assert"
            | "assert_eq"
            | "assert_ne"
            | "assert_relative_eq"
            | "dbg"
            | "debug_assert"
            | "debug_assert_eq"
            | "debug_assert_ne"
            | "debug"
            | "eprint"
            | "eprintln"
            | "format"
            | "format_args"
            | "is_aarch64_feature_detected"
            | "is_arm_feature_detected"
            | "is_x86_feature_detected"
            | "info"
            | "json"
            | "matches"
            | "panic"
            | "params"
            | "print"
            | "println"
            | "prop_assert"
            | "proptest"
            | "todo"
            | "thread_local"
            | "unimplemented"
            | "unreachable"
            | "value_parser"
            | "vec"
            | "warn"
            | "write"
            | "writeln"
    ) {
        return macro_tokens_name_protected_work(&mac.tokens).then(|| {
            format!("unclassifiable protected-work macro `{label}!` in test-bearing scope")
        });
    }
    Some(format!(
        "unclassifiable macro invocation `{label}!` in test-bearing scope"
    ))
}

struct FunctionMacroCollector {
    error: Option<String>,
}

impl<'ast> Visit<'ast> for FunctionMacroCollector {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if self.error.is_none() {
            self.error = unclassifiable_macro(mac);
        }
    }
}

struct ItemMacroCollector {
    error: Option<String>,
}

struct MacroCollector<'ast> {
    macros: Vec<&'ast syn::Macro>,
}

impl<'ast> Visit<'ast> for MacroCollector<'ast> {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.macros.push(mac);
    }
}

fn macro_delimiter(mac: &syn::Macro) -> (char, usize, char, usize) {
    match &mac.delimiter {
        syn::MacroDelimiter::Paren(token) => (
            '(',
            token.span.open().byte_range().start,
            ')',
            token.span.close().byte_range().start,
        ),
        syn::MacroDelimiter::Brace(token) => (
            '{',
            token.span.open().byte_range().start,
            '}',
            token.span.close().byte_range().start,
        ),
        syn::MacroDelimiter::Bracket(token) => (
            '[',
            token.span.open().byte_range().start,
            ']',
            token.span.close().byte_range().start,
        ),
    }
}

impl<'ast> Visit<'ast> for ItemMacroCollector {
    fn visit_item_fn(&mut self, _function: &'ast ItemFn) {}

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        if item.ident.is_none() && self.error.is_none() {
            self.error = unclassifiable_macro(&item.mac);
        }
    }
}

fn unclassifiable_item_macro(file: &syn::File) -> Option<String> {
    let mut collector = ItemMacroCollector { error: None };
    collector.visit_file(file);
    collector.error
}

fn unclassifiable_include(file: &syn::File, test_cfg: bool) -> Option<String> {
    let mut collector = MacroCollector { macros: Vec::new() };
    collector.visit_file(file);
    collector.macros.into_iter().find_map(|mac| {
        mac.path.is_ident("include").then(|| {
            if test_cfg {
                "unclassifiable include! in test-bearing scope".to_string()
            } else {
                "unclassifiable include! in Metal hazard scope".to_string()
            }
        })
    })
}

fn module_functions<'ast>(items: &'ast [syn::Item], functions: &mut Vec<&'ast ItemFn>) {
    for item in items {
        match item {
            syn::Item::Fn(function) => functions.push(function),
            syn::Item::Mod(module) => {
                if let Some((_, contents)) = &module.content {
                    module_functions(contents, functions);
                }
            }
            _ => {}
        }
    }
}

fn inline_modules<'ast>(items: &'ast [syn::Item], modules: &mut Vec<&'ast syn::ItemMod>) {
    for item in items {
        if let syn::Item::Mod(module) = item
            && let Some((_, contents)) = &module.content
        {
            modules.push(module);
            inline_modules(contents, modules);
        }
    }
}

/// The receiver type name for an `impl` block, used as a path segment so
/// methods on different types with the same name (e.g. two `fn load`) do not
/// collide in a construction site's stable key.
fn impl_self_type_label(self_ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = self_ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident.to_string();
    }
    "impl".to_string()
}

#[derive(Clone)]
struct ModuleSource {
    path: PathBuf,
    external_cfg: CfgFormula,
}

fn syn_meta_arguments(list: &syn::MetaList, context: &str) -> Result<Vec<Meta>, String> {
    Punctuated::<Meta, Token![,]>::parse_terminated
        .parse2(list.tokens.clone())
        .map(|arguments| arguments.into_iter().collect())
        .map_err(|reason| format!("{context}: unclassifiable cfg syntax: {reason}"))
}

fn syn_cfg_predicate(meta: &Meta, test_cfg: bool, context: &str) -> Result<CfgFormula, String> {
    match meta {
        Meta::Path(path) => {
            let label = path_label(path);
            Ok(match label.as_str() {
                "test" if test_cfg => CfgFormula::True,
                "test" => CfgFormula::False,
                "unix" => CfgFormula::True,
                "windows" => CfgFormula::False,
                _ => CfgFormula::Atom(label),
            })
        }
        Meta::NameValue(value) => {
            let label = path_label(&value.path);
            let syn::Expr::Lit(expression) = &value.value else {
                return Err(format!(
                    "{context}: cfg value for `{label}` is not a literal"
                ));
            };
            let syn::Lit::Str(value) = &expression.lit else {
                return Err(format!(
                    "{context}: cfg value for `{label}` is not a string"
                ));
            };
            let value = value.value();
            Ok(match (label.as_str(), value.as_str()) {
                ("target_os", "macos") | ("target_family", "unix") => CfgFormula::True,
                ("target_os", _) | ("target_family", "windows") => CfgFormula::False,
                ("feature", "metal-gpu") => CfgFormula::True,
                _ => CfgFormula::Atom(format!("{label}={value}")),
            })
        }
        Meta::List(list) => {
            let name = path_label(&list.path);
            let arguments = syn_meta_arguments(list, context)?;
            match name.as_str() {
                "all" => Ok(CfgFormula::all(
                    arguments
                        .iter()
                        .map(|argument| syn_cfg_predicate(argument, test_cfg, context))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "any" => Ok(CfgFormula::any(
                    arguments
                        .iter()
                        .map(|argument| syn_cfg_predicate(argument, test_cfg, context))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "not" if arguments.len() == 1 => Ok(CfgFormula::not(syn_cfg_predicate(
                    &arguments[0],
                    test_cfg,
                    context,
                )?)),
                "not" => Err(format!("{context}: cfg(not(...)) needs one predicate")),
                _ => Ok(CfgFormula::Atom(name)),
            }
        }
    }
}

fn syn_cfg_effect(meta: &Meta, test_cfg: bool, context: &str) -> Result<CfgFormula, String> {
    let Meta::List(list) = meta else {
        let name = path_label(meta.path());
        return if matches!(name.as_str(), "cfg" | "cfg_attr") {
            Err(format!("{context}: malformed `{name}` attribute"))
        } else {
            Ok(CfgFormula::True)
        };
    };
    let name = path_label(&list.path);
    let arguments = syn_meta_arguments(list, context)?;
    if name == "cfg" {
        if arguments.len() != 1 {
            return Err(format!("{context}: cfg attribute needs one predicate"));
        }
        return syn_cfg_predicate(&arguments[0], test_cfg, context);
    }
    if name != "cfg_attr" {
        return Ok(CfgFormula::True);
    }
    if arguments.len() < 2 {
        return Err(format!(
            "{context}: cfg_attr attribute needs a predicate and emitted attribute"
        ));
    }
    let predicate = syn_cfg_predicate(&arguments[0], test_cfg, context)?;
    Ok(CfgFormula::all(
        arguments
            .iter()
            .skip(1)
            .map(|emitted| {
                Ok(CfgFormula::any([
                    CfgFormula::not(predicate.clone()),
                    syn_cfg_effect(emitted, test_cfg, context)?,
                ]))
            })
            .collect::<Result<Vec<_>, String>>()?,
    ))
}

fn syn_attributes_formula(
    attributes: &[Attribute],
    test_cfg: bool,
    context: &str,
) -> Result<CfgFormula, String> {
    Ok(CfgFormula::all(
        attributes
            .iter()
            .map(|attribute| syn_cfg_effect(&attribute.meta, test_cfg, context))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn cfg_attr_emits_path(meta: &Meta, context: &str) -> Result<bool, String> {
    let Meta::List(list) = meta else {
        return if meta.path().is_ident("cfg_attr") {
            Err(format!("{context}: malformed `cfg_attr` attribute"))
        } else {
            Ok(false)
        };
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(false);
    }
    for emitted in syn_meta_arguments(list, context)?.iter().skip(1) {
        if emitted.path().is_ident("path") || cfg_attr_emits_path(emitted, context)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn literal_module_path(attributes: &[Attribute], context: &str) -> Result<Option<PathBuf>, String> {
    for attribute in attributes {
        if cfg_attr_emits_path(&attribute.meta, context)? {
            return Err(format!(
                "{context}: unclassifiable cfg_attr-generated module path"
            ));
        }
        if !attribute.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(value) = &attribute.meta else {
            return Err(format!("{context}: unclassifiable module path attribute"));
        };
        let syn::Expr::Lit(expression) = &value.value else {
            return Err(format!("{context}: module path is not a string literal"));
        };
        let syn::Lit::Str(path) = &expression.lit else {
            return Err(format!("{context}: module path is not a string literal"));
        };
        return Ok(Some(PathBuf::from(path.value())));
    }
    Ok(None)
}

struct ModuleGraph<'a> {
    manifest_dir: &'a Path,
    manifest_canonical: PathBuf,
    test_cfg: bool,
    sources: BTreeMap<PathBuf, CfgFormula>,
    visiting: BTreeSet<PathBuf>,
}

impl ModuleGraph<'_> {
    fn load(
        &mut self,
        path: &Path,
        module_dir: &Path,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        let canonical = std::fs::canonicalize(path).map_err(|reason| {
            format!(
                "unclassifiable compiler-selected module {}: {reason}",
                path.display()
            )
        })?;
        if !canonical.starts_with(&self.manifest_canonical) {
            return Err(format!(
                "unclassifiable compiler-selected module outside crate boundary: {}",
                canonical.display()
            ));
        }
        if !self.visiting.insert(canonical.clone()) {
            return Err(format!(
                "module graph cycle while classifying {}",
                canonical.display()
            ));
        }
        let source = std::fs::read_to_string(&canonical).map_err(|reason| {
            format!(
                "could not read compiler-selected module {}: {reason}",
                canonical.display()
            )
        })?;
        let context = canonical
            .strip_prefix(self.manifest_dir)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();
        let syntax = syn::parse_file(&source).map_err(|reason| {
            format!("{context}: Rust syntax could not be classified: {reason}")
        })?;
        let file_cfg = CfgFormula::all([
            inherited_cfg,
            syn_attributes_formula(&syntax.attrs, self.test_cfg, &context)?,
        ]);
        if !file_cfg.satisfiable()? {
            self.visiting.remove(&canonical);
            return Ok(());
        }
        if let Some(existing) = self.sources.get_mut(&canonical) {
            *existing = CfgFormula::any([existing.clone(), file_cfg]);
            self.visiting.remove(&canonical);
            return Ok(());
        }
        self.sources.insert(canonical.clone(), file_cfg.clone());
        self.walk_items(&syntax.items, &canonical, module_dir, file_cfg)?;
        self.visiting.remove(&canonical);
        Ok(())
    }

    fn walk_items(
        &mut self,
        items: &[syn::Item],
        source_path: &Path,
        module_dir: &Path,
        inherited_cfg: CfgFormula,
    ) -> Result<(), String> {
        let context = source_path
            .strip_prefix(self.manifest_dir)
            .unwrap_or(source_path)
            .to_string_lossy()
            .into_owned();
        for item in items {
            let syn::Item::Mod(module) = item else {
                continue;
            };
            let module_cfg = CfgFormula::all([
                inherited_cfg.clone(),
                syn_attributes_formula(&module.attrs, self.test_cfg, &context)?,
            ]);
            if !module_cfg.satisfiable()? {
                continue;
            }
            if let Some((_, contents)) = &module.content {
                if literal_module_path(&module.attrs, &context)?.is_some() {
                    return Err(format!(
                        "{context}: unclassifiable path attribute on inline module `{}`",
                        module.ident
                    ));
                }
                self.walk_items(
                    contents,
                    source_path,
                    &module_dir.join(module.ident.to_string()),
                    module_cfg,
                )?;
                continue;
            }
            let direct_path = literal_module_path(&module.attrs, &context)?;
            let target = if let Some(relative) = direct_path {
                module_dir.join(relative)
            } else {
                let flat = module_dir.join(format!("{}.rs", module.ident));
                let nested = module_dir.join(module.ident.to_string()).join("mod.rs");
                match (flat.exists(), nested.exists()) {
                    (true, false) => flat,
                    (false, true) => nested,
                    (true, true) => {
                        return Err(format!(
                            "{context}: ambiguous compiler-selected module `{}`",
                            module.ident
                        ));
                    }
                    (false, false) => {
                        return Err(format!(
                            "{context}: compiler-selected module `{}` could not be resolved",
                            module.ident
                        ));
                    }
                }
            };
            let child_module_dir = if target.file_name().is_some_and(|name| name == "mod.rs") {
                target.parent().unwrap_or(module_dir).to_path_buf()
            } else {
                target
                    .parent()
                    .unwrap_or(module_dir)
                    .join(module.ident.to_string())
            };
            self.load(&target, &child_module_dir, module_cfg)?;
        }
        Ok(())
    }
}

/// Returns the compiler-selected source closure for one Cargo target root.
///
/// The serialization boundary is bounded repository-owned tests, benches,
/// examples, and measurement binaries that can submit Metal work, not a
/// physical `src/**/*.rs` walk. Production modules are covered through those
/// callers because they cannot execute independently. Long-running interactive
/// and service targets are explicit exemptions: they must acquire the same
/// fleet lock externally when used as measurements. The crate has no build
/// script or executable Metal doctest, so neither is a current hazard entrypoint.
fn module_source_closure(
    manifest_dir: &Path,
    root: &Path,
    test_cfg: bool,
) -> Result<Vec<ModuleSource>, String> {
    let manifest_canonical = std::fs::canonicalize(manifest_dir).map_err(|reason| {
        format!(
            "could not resolve crate boundary {}: {reason}",
            manifest_dir.display()
        )
    })?;
    let mut graph = ModuleGraph {
        manifest_dir,
        manifest_canonical,
        test_cfg,
        sources: BTreeMap::new(),
        visiting: BTreeSet::new(),
    };
    let module_dir = root
        .parent()
        .ok_or_else(|| format!("target root {} has no parent", root.display()))?;
    graph.load(root, module_dir, CfgFormula::True)?;
    Ok(graph
        .sources
        .into_iter()
        .map(|(path, external_cfg)| ModuleSource { path, external_cfg })
        .collect())
}

struct ParsedModuleSource {
    relative: String,
    parsed: StructuredSource,
}

fn parsed_module_closure(
    manifest_dir: &Path,
    root: &Path,
    test_cfg: bool,
) -> Result<Vec<ParsedModuleSource>, String> {
    module_source_closure(manifest_dir, root, test_cfg)?
        .into_iter()
        .map(|module| {
            let source = std::fs::read_to_string(&module.path).map_err(|reason| {
                format!("could not read module {}: {reason}", module.path.display())
            })?;
            let relative = module
                .path
                .strip_prefix(manifest_dir)
                .unwrap_or(&module.path)
                .to_string_lossy()
                .into_owned();
            let parsed =
                StructuredSource::parse_module_source(manifest_dir, &module, &source, test_cfg)?;
            Ok(ParsedModuleSource { relative, parsed })
        })
        .collect()
}

impl CallSelector {
    fn label(self) -> String {
        match self {
            Self::Path(path) => format!("{}()", path.join("::")),
            Self::Method(method) => format!(".{method}()"),
        }
    }

    fn final_name(self) -> &'static str {
        match self {
            Self::Path(path) => path.last().copied().unwrap_or(""),
            Self::Method(method) => method,
        }
    }
}

impl StructuredSource {
    fn parse(context: &str, source: &str, test_cfg: bool) -> Result<Self, String> {
        let syntax = syn::parse_file(source).map_err(|reason| {
            format!("{context}: Rust syntax could not be classified: {reason}")
        })?;
        let tokens = rust_tokens(source).map_err(|reason| format!("{context}: {reason}"))?;
        let (pairs, parents) =
            delimiter_structure(&tokens).map_err(|reason| format!("{context}: {reason}"))?;
        let mut parsed = Self {
            context: context.to_string(),
            source: source.to_string(),
            tokens,
            pairs,
            parents,
            macro_ranges: Vec::new(),
            functions: Vec::new(),
            function_paths: Vec::new(),
            scopes: Vec::new(),
            file_attributes: Vec::new(),
            test_cfg,
            external_cfg: CfgFormula::True,
            unclassifiable_test_scope: None,
        };
        parsed.macro_ranges = parsed.find_macro_ranges(&syntax)?;
        parsed.functions = parsed.find_functions(&syntax)?;
        parsed.function_paths = parsed.find_function_paths(&syntax)?;
        parsed.scopes = parsed.find_scopes(&syntax)?;
        parsed.file_attributes = parsed.inner_attributes_at(0).0;
        parsed.unclassifiable_test_scope = unclassifiable_include(&syntax, test_cfg);
        if test_cfg && parsed.unclassifiable_test_scope.is_none() {
            parsed.unclassifiable_test_scope = unclassifiable_item_macro(&syntax);
        }
        Ok(parsed)
    }

    fn find_macro_ranges(&self, syntax: &syn::File) -> Result<Vec<Range<usize>>, String> {
        let mut collector = MacroCollector { macros: Vec::new() };
        collector.visit_file(syntax);
        let mut ranges = Vec::new();
        for mac in collector.macros {
            let (open, open_offset, close, close_offset) = macro_delimiter(mac);
            let opening = self
                .tokens
                .iter()
                .position(|token| token.offset == open_offset && token.is_punct(open))
                .ok_or_else(|| {
                    format!(
                        "{}: macro opening delimiter could not be mapped",
                        self.context
                    )
                })?;
            let closing = self
                .tokens
                .iter()
                .position(|token| token.offset == close_offset && token.is_punct(close))
                .ok_or_else(|| {
                    format!(
                        "{}: macro closing delimiter could not be mapped",
                        self.context
                    )
                })?;
            if self.pairs[opening] != Some(closing) {
                return Err(format!(
                    "{}: parsed macro delimiters do not match lexical delimiters",
                    self.context
                ));
            }
            ranges.push(opening..closing + 1);
        }
        Ok(ranges)
    }

    fn in_macro(&self, index: usize) -> bool {
        self.macro_ranges.iter().any(|range| range.contains(&index))
    }

    fn attribute_at(&self, start: usize, inner: bool) -> Option<(AttributeSpec, usize)> {
        if !self.tokens.get(start)?.is_punct('#') {
            return None;
        }
        let opening = if inner {
            if !self.tokens.get(start + 1)?.is_punct('!') {
                return None;
            }
            start + 2
        } else {
            start + 1
        };
        if !self.tokens.get(opening)?.is_punct('[') {
            return None;
        }
        let closing = self.pairs.get(opening).and_then(|pair| *pair)?;
        Some((
            AttributeSpec {
                content: opening + 1..closing,
            },
            closing + 1,
        ))
    }

    fn inner_attributes_at(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while let Some((attribute, next)) = self.attribute_at(cursor, true) {
            attributes.push(attribute);
            cursor = next;
        }
        (attributes, cursor)
    }

    fn outer_attributes_at(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while let Some((attribute, next)) = self.attribute_at(cursor, false) {
            attributes.push(attribute);
            cursor = next;
        }
        (attributes, cursor)
    }

    fn outer_attributes_before(&self, mut cursor: usize) -> (Vec<AttributeSpec>, usize) {
        let mut attributes = Vec::new();
        while cursor > 0 && self.tokens[cursor - 1].is_punct(']') {
            let closing = cursor - 1;
            let Some(opening) = self.pairs[closing] else {
                break;
            };
            let Some(hash) = opening.checked_sub(1) else {
                break;
            };
            if !self.tokens[hash].is_punct('#') || hash > 0 && self.tokens[hash - 1].is_punct('!') {
                break;
            }
            attributes.push(AttributeSpec {
                content: opening + 1..closing,
            });
            cursor = hash;
        }
        attributes.reverse();
        (attributes, cursor)
    }

    fn item_attributes_before(&self, item: usize) -> Vec<AttributeSpec> {
        let mut cursor = item;
        loop {
            let Some(previous) = cursor.checked_sub(1) else {
                break;
            };
            if self.tokens[previous].ident().is_some_and(|ident| {
                matches!(
                    ident,
                    "async" | "const" | "default" | "extern" | "pub" | "unsafe"
                )
            }) {
                cursor = previous;
                continue;
            }
            if matches!(self.tokens[previous].kind, RustTokenKind::String(_))
                && previous > 0
                && self.tokens[previous - 1].ident() == Some("extern")
            {
                cursor = previous - 1;
                continue;
            }
            if self.tokens[previous].is_punct(')')
                && let Some(opening) = self.pairs[previous]
                && opening > 0
                && self.tokens[opening - 1].ident() == Some("pub")
            {
                cursor = opening - 1;
                continue;
            }
            break;
        }
        self.outer_attributes_before(cursor).0
    }

    fn find_functions(&self, syntax: &syn::File) -> Result<Vec<FunctionSpec>, String> {
        let mut collected = Vec::new();
        module_functions(&syntax.items, &mut collected);
        let mut functions = Vec::new();
        for function in collected {
            let name = function.sig.ident.to_string();
            let fn_offset = function.sig.fn_token.span.byte_range().start;
            let item = self
                .tokens
                .iter()
                .position(|token| token.offset == fn_offset && token.ident() == Some("fn"))
                .ok_or_else(|| {
                    format!(
                        "{}: function `{name}` could not be mapped to parsed source",
                        self.context
                    )
                })?;
            let opening_offset = function.block.brace_token.span.open().byte_range().start;
            let opening = self
                .tokens
                .iter()
                .position(|token| token.offset == opening_offset && token.is_punct('{'))
                .ok_or_else(|| {
                    format!(
                        "{}: function `{name}` body could not be mapped to parsed source",
                        self.context
                    )
                })?;
            let closing = self.pairs[opening].ok_or_else(|| {
                format!(
                    "{}: function `{name}` has no brace-balanced body",
                    self.context
                )
            })?;
            let mut macros = FunctionMacroCollector { error: None };
            macros.visit_block(&function.block);
            functions.push(FunctionSpec {
                name,
                body: opening + 1..closing,
                returns_unit: function.sig.asyncness.is_none()
                    && (matches!(&function.sig.output, syn::ReturnType::Default)
                        || matches!(&function.sig.output, syn::ReturnType::Type(_, output)
                        if matches!(&**output, syn::Type::Tuple(tuple) if tuple.elems.is_empty()))),
                attributes: self.item_attributes_before(item),
                test_registration: classify_test_attributes(&function.attrs),
                unclassifiable_macro: macros.error,
            });
        }
        Ok(functions)
    }

    /// Maps a `syn::Block`'s brace pair to the token-index range between
    /// them, the same convention `FunctionSpec::body` uses.
    fn token_range_for_block(
        &self,
        block: &syn::Block,
        label: &str,
    ) -> Result<Range<usize>, String> {
        let opening_offset = block.brace_token.span.open().byte_range().start;
        let opening = self
            .tokens
            .iter()
            .position(|token| token.offset == opening_offset && token.is_punct('{'))
            .ok_or_else(|| {
                format!(
                    "{}: function `{label}` body could not be mapped to parsed source",
                    self.context
                )
            })?;
        let closing = self.pairs[opening].ok_or_else(|| {
            format!(
                "{}: function `{label}` has no brace-balanced body",
                self.context
            )
        })?;
        Ok(opening + 1..closing)
    }

    fn collect_function_paths(
        &self,
        items: &[syn::Item],
        path: &mut Vec<String>,
        out: &mut Vec<(Range<usize>, String)>,
    ) -> Result<(), String> {
        for item in items {
            match item {
                syn::Item::Fn(function) => {
                    path.push(function.sig.ident.to_string());
                    let label = path.join("::");
                    out.push((self.token_range_for_block(&function.block, &label)?, label));
                    path.pop();
                }
                syn::Item::Mod(module) => {
                    if let Some((_, contents)) = &module.content {
                        path.push(module.ident.to_string());
                        self.collect_function_paths(contents, path, out)?;
                        path.pop();
                    }
                }
                syn::Item::Impl(item_impl) => {
                    path.push(impl_self_type_label(&item_impl.self_ty));
                    for impl_item in &item_impl.items {
                        if let syn::ImplItem::Fn(method) = impl_item {
                            path.push(method.sig.ident.to_string());
                            let label = path.join("::");
                            out.push((self.token_range_for_block(&method.block, &label)?, label));
                            path.pop();
                        }
                    }
                    path.pop();
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Every free function and `impl` method in the file, keyed by its
    /// token-index body range and named by the `mod`/`impl` path enclosing
    /// it (e.g. `MetalChatBackend::load`). Construction sites are named by
    /// looking up which of these ranges contains them, rather than by their
    /// source line and column, so a site's identity survives edits that
    /// shift line numbers without changing the program structure (#1357).
    fn find_function_paths(
        &self,
        syntax: &syn::File,
    ) -> Result<Vec<(Range<usize>, String)>, String> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.collect_function_paths(&syntax.items, &mut path, &mut out)?;
        Ok(out)
    }

    /// The innermost `mod`/`impl`/`fn` path enclosing `token`, or
    /// `"<module scope>"` when the token sits outside every recorded
    /// function (e.g. in a `static` initializer).
    fn enclosing_function_path(&self, token: usize) -> String {
        self.function_paths
            .iter()
            .filter(|(range, _)| range.contains(&token))
            .min_by_key(|(range, _)| range.end.saturating_sub(range.start))
            .map(|(_, path)| path.clone())
            .unwrap_or_else(|| "<module scope>".to_string())
    }

    fn find_scopes(&self, syntax: &syn::File) -> Result<Vec<ScopeSpec>, String> {
        let mut scopes = self
            .functions
            .iter()
            .map(|function| ScopeSpec {
                body: function.body.clone(),
                attributes: function.attributes.clone(),
            })
            .collect::<Vec<_>>();
        let mut modules = Vec::new();
        inline_modules(&syntax.items, &mut modules);
        for module in modules {
            let Some((brace, _)) = &module.content else {
                continue;
            };
            let item_offset = module.mod_token.span.byte_range().start;
            let item = self
                .tokens
                .iter()
                .position(|token| token.offset == item_offset && token.ident() == Some("mod"))
                .ok_or_else(|| {
                    format!(
                        "{}: inline module `{}` could not be mapped to parsed source",
                        self.context, module.ident
                    )
                })?;
            let opening_offset = brace.span.open().byte_range().start;
            let opening = self
                .tokens
                .iter()
                .position(|token| token.offset == opening_offset && token.is_punct('{'))
                .ok_or_else(|| {
                    format!(
                        "{}: inline module `{}` body could not be mapped",
                        self.context, module.ident
                    )
                })?;
            let closing = self.pairs[opening]
                .ok_or_else(|| format!("{}: inline module has no closing brace", self.context))?;
            scopes.push(ScopeSpec {
                body: opening + 1..closing,
                attributes: self.item_attributes_before(item),
            });
        }
        for opening in 0..self.tokens.len() {
            if !self.tokens[opening].is_punct('{') {
                continue;
            }
            let Some(closing) = self.pairs[opening] else {
                continue;
            };
            let enclosing = self.construct_attributes(opening);
            if !enclosing.is_empty() {
                scopes.push(ScopeSpec {
                    body: opening + 1..closing,
                    attributes: enclosing,
                });
            }
            let inner = self.inner_attributes_at(opening + 1).0;
            if !inner.is_empty() {
                scopes.push(ScopeSpec {
                    body: opening + 1..closing,
                    attributes: inner,
                });
            }
        }
        Ok(scopes)
    }

    fn construct_attributes(&self, opening: usize) -> Vec<AttributeSpec> {
        let parent = self.parents[opening];
        let initial = parent.map_or(0, |parent| parent + 1);
        let start = (initial..opening)
            .rev()
            .find(|&index| {
                self.parents[index] == parent
                    && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            })
            .map_or(initial, |index| index + 1);
        self.outer_attributes_at(start).0
    }

    fn split_arguments(&self, range: Range<usize>) -> Vec<Range<usize>> {
        let mut arguments = Vec::new();
        let mut start = range.start;
        let mut stack = Vec::<char>::new();
        for index in range.clone() {
            match self.tokens[index].kind {
                RustTokenKind::Punct(open @ ('(' | '[' | '{')) => stack.push(open),
                RustTokenKind::Punct(')' | ']' | '}') => {
                    stack.pop();
                }
                RustTokenKind::Punct(',') if stack.is_empty() => {
                    arguments.push(start..index);
                    start = index + 1;
                }
                _ => {}
            }
        }
        arguments.push(start..range.end);
        arguments
    }

    fn checked_arguments(
        &self,
        range: Range<usize>,
        label: &str,
    ) -> Result<Vec<Range<usize>>, String> {
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let trailing_comma = self.tokens[range.end - 1].is_punct(',');
        let mut arguments = self.split_arguments(range);
        if trailing_comma && arguments.last().is_some_and(Range::is_empty) {
            arguments.pop();
        }
        if arguments.iter().any(Range::is_empty) {
            return Err(format!("{}: empty {label} argument", self.context));
        }
        Ok(arguments)
    }

    fn normalized_tokens(&self, range: Range<usize>) -> String {
        let mut normalized = String::new();
        for token in &self.tokens[range] {
            match &token.kind {
                RustTokenKind::Ident { text, raw } => {
                    if *raw {
                        normalized.push_str("r#");
                    }
                    normalized.push_str(text);
                }
                RustTokenKind::String(value) => {
                    normalized.push('"');
                    normalized.push_str(value);
                    normalized.push('"');
                }
                RustTokenKind::Punct(punct) => normalized.push(*punct),
                RustTokenKind::Other => normalized.push('?'),
            }
        }
        normalized
    }

    fn cfg_predicate(&self, range: Range<usize>) -> Result<CfgFormula, String> {
        if range.is_empty() {
            return Err(format!("{}: empty cfg predicate", self.context));
        }
        if let Some(name) = self.tokens[range.start].ident()
            && range.start + 1 < range.end
            && self.tokens[range.start + 1].is_punct('(')
            && self.pairs[range.start + 1] == Some(range.end - 1)
        {
            let arguments =
                self.checked_arguments(range.start + 2..range.end - 1, "cfg predicate")?;
            return match name {
                "all" => Ok(CfgFormula::all(
                    arguments
                        .into_iter()
                        .map(|argument| self.cfg_predicate(argument))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "any" => Ok(CfgFormula::any(
                    arguments
                        .into_iter()
                        .map(|argument| self.cfg_predicate(argument))
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                "not" if arguments.len() == 1 => {
                    Ok(CfgFormula::not(self.cfg_predicate(arguments[0].clone())?))
                }
                "not" => Err(format!(
                    "{}: cfg(not(...)) needs one predicate",
                    self.context
                )),
                _ => Ok(CfgFormula::Atom(self.normalized_tokens(range))),
            };
        }
        if range.len() == 1 {
            let Some(name) = self.tokens[range.start].ident() else {
                return Err(format!(
                    "{}: unsupported cfg predicate `{}`",
                    self.context,
                    self.normalized_tokens(range)
                ));
            };
            return Ok(match name {
                "test" => {
                    if self.test_cfg {
                        CfgFormula::True
                    } else {
                        CfgFormula::False
                    }
                }
                "unix" => CfgFormula::True,
                "windows" => CfgFormula::False,
                _ => CfgFormula::Atom(name.to_string()),
            });
        }
        if range.len() == 3 && self.tokens[range.start + 1].is_punct('=') {
            let Some(key) = self.tokens[range.start].ident() else {
                return Err(format!(
                    "{}: unsupported cfg key `{}`",
                    self.context,
                    self.normalized_tokens(range)
                ));
            };
            let RustTokenKind::String(value) = &self.tokens[range.start + 2].kind else {
                return Err(format!(
                    "{}: cfg value for `{key}` is not a string",
                    self.context
                ));
            };
            return Ok(match (key, value.as_str()) {
                ("target_os", "macos") | ("target_family", "unix") => CfgFormula::True,
                ("target_os", _) | ("target_family", "windows") => CfgFormula::False,
                ("feature", "metal-gpu") => CfgFormula::True,
                _ => CfgFormula::Atom(format!("{key}={value}")),
            });
        }
        Err(format!(
            "{}: unsupported cfg predicate `{}`",
            self.context,
            self.normalized_tokens(range)
        ))
    }

    fn meta_cfg_effect(&self, range: Range<usize>) -> Result<CfgFormula, String> {
        if range.is_empty() {
            return Err(format!("{}: empty cfg attribute argument", self.context));
        }
        let Some(name) = self.tokens.get(range.start).and_then(RustToken::ident) else {
            return Ok(CfgFormula::True);
        };
        if !matches!(name, "cfg" | "cfg_attr") {
            return Ok(CfgFormula::True);
        }
        let opening = range.start + 1;
        if !self
            .tokens
            .get(opening)
            .is_some_and(|token| token.is_punct('('))
            || self.pairs.get(opening).and_then(|pair| *pair) != Some(range.end - 1)
        {
            return Err(format!("{}: malformed `{name}` attribute", self.context));
        }
        let arguments = self.checked_arguments(opening + 1..range.end - 1, "cfg attribute")?;
        if name == "cfg" {
            if arguments.len() != 1 {
                return Err(format!(
                    "{}: cfg attribute needs one predicate",
                    self.context
                ));
            }
            return self.cfg_predicate(arguments[0].clone());
        }
        if arguments.len() < 2 {
            return Err(format!(
                "{}: cfg_attr attribute needs a predicate and emitted attribute",
                self.context
            ));
        }
        let predicate = self.cfg_predicate(arguments[0].clone())?;
        let mut effects = Vec::new();
        for emitted in arguments.into_iter().skip(1) {
            let emitted_effect = self.meta_cfg_effect(emitted)?;
            effects.push(CfgFormula::any([
                CfgFormula::not(predicate.clone()),
                emitted_effect,
            ]));
        }
        Ok(CfgFormula::all(effects))
    }

    fn attributes_formula(&self, attributes: &[AttributeSpec]) -> Result<CfgFormula, String> {
        Ok(CfgFormula::all(
            attributes
                .iter()
                .map(|attribute| self.meta_cfg_effect(attribute.content.clone()))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    fn nearest_brace(&self, index: usize) -> Option<usize> {
        let mut parent = self.parents.get(index).and_then(|parent| *parent);
        while let Some(opening) = parent {
            if self.tokens[opening].is_punct('{') {
                return Some(opening);
            }
            parent = self.parents[opening];
        }
        None
    }

    fn statement_attributes(&self, site: usize) -> Vec<AttributeSpec> {
        let Some(block) = self.nearest_brace(site) else {
            return Vec::new();
        };
        let mut start = block + 1;
        for index in block + 1..site {
            let same_block =
                self.nearest_brace(index) == Some(block) || self.parents[index] == Some(block);
            if same_block && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            {
                start = index + 1;
            }
        }
        self.outer_attributes_at(start).0
    }

    fn formula_at(&self, site: usize) -> Result<CfgFormula, String> {
        let mut formulas = vec![self.external_cfg.clone()];
        formulas.push(self.attributes_formula(&self.file_attributes)?);
        for scope in &self.scopes {
            if scope.body.contains(&site) {
                formulas.push(self.attributes_formula(&scope.attributes)?);
            }
        }
        formulas.push(self.attributes_formula(&self.statement_attributes(site))?);
        Ok(CfgFormula::all(formulas))
    }

    fn path_start(&self, final_index: usize, path: &[&str]) -> Option<usize> {
        if self.tokens.get(final_index)?.ident() != path.last().copied() {
            return None;
        }
        let mut cursor = final_index;
        for segment in path[..path.len().saturating_sub(1)].iter().rev() {
            if cursor < 3
                || !self.tokens[cursor - 1].is_punct(':')
                || !self.tokens[cursor - 2].is_punct(':')
                || self.tokens[cursor - 3].ident() != Some(*segment)
            {
                return None;
            }
            cursor -= 3;
        }
        Some(cursor)
    }

    fn selector_candidate(&self, index: usize, selector: CallSelector) -> Option<usize> {
        match selector {
            CallSelector::Path(path) => {
                let start = self.path_start(index, path)?;
                if path.len() == 1
                    && start > 0
                    && (self.tokens[start - 1].is_punct('.')
                        || self.tokens[start - 1].is_punct(':'))
                {
                    return None;
                }
                Some(start)
            }
            CallSelector::Method(method) => {
                if self.tokens[index].ident() == Some(method)
                    && index > 0
                    && self.tokens[index - 1].is_punct('.')
                {
                    Some(index)
                } else {
                    None
                }
            }
        }
    }

    fn call_sites(
        &self,
        range: Range<usize>,
        selector: CallSelector,
        fail_closed: bool,
    ) -> Result<Vec<usize>, String> {
        let mut calls = Vec::new();
        for index in range {
            if self.tokens[index].ident() != Some(selector.final_name()) {
                continue;
            }
            let Some(start) = self.selector_candidate(index, selector) else {
                if fail_closed && matches!(selector, CallSelector::Method(_)) {
                    return Err(format!(
                        "{}: unclassifiable protected-work syntax near `{}`",
                        self.context,
                        selector.label()
                    ));
                }
                continue;
            };
            if matches!(selector, CallSelector::Path(path) if path.len() == 1)
                && start > 0
                && self.tokens[start - 1].ident() == Some("fn")
            {
                continue;
            }
            if self.in_macro(index) {
                if fail_closed {
                    return Err(format!(
                        "{}: unclassifiable protected-work syntax containing `{}`",
                        self.context,
                        selector.label()
                    ));
                }
                continue;
            }
            if self
                .tokens
                .get(index + 1)
                .is_some_and(|token| token.is_punct('('))
            {
                calls.push(index);
            } else if fail_closed {
                return Err(format!(
                    "{}: unclassifiable protected-work syntax near `{}`",
                    self.context,
                    selector.label()
                ));
            }
        }
        Ok(calls)
    }

    fn has_selector(&self, selector: CallSelector) -> Result<bool, String> {
        if let Some(reason) = &self.unclassifiable_test_scope {
            return Err(format!("{}: {reason}", self.context));
        }
        self.reject_selector_in_macros(selector)?;
        Ok(!self
            .call_sites(0..self.tokens.len(), selector, false)?
            .is_empty())
    }

    fn reject_selector_in_macros(&self, selector: CallSelector) -> Result<(), String> {
        for range in &self.macro_ranges {
            for index in range.clone() {
                if self.tokens[index].ident() == Some(selector.final_name())
                    && self.selector_candidate(index, selector).is_some()
                {
                    return Err(format!(
                        "{}: unclassifiable protected-work syntax containing `{}`",
                        self.context,
                        selector.label()
                    ));
                }
            }
        }
        Ok(())
    }

    fn has_code_identifier(&self, identifier: &str) -> bool {
        self.tokens
            .iter()
            .enumerate()
            .any(|(index, token)| token.ident() == Some(identifier) && !self.in_macro(index))
    }

    fn test_functions(&self) -> Result<Vec<&FunctionSpec>, String> {
        if let Some(reason) = &self.unclassifiable_test_scope {
            return Err(format!("{}: {reason}", self.context));
        }
        let mut functions = Vec::new();
        for function in &self.functions {
            match &function.test_registration {
                TestRegistration::No => continue,
                TestRegistration::Unclassifiable(attribute) => {
                    return Err(format!(
                        "{}: unclassifiable test function attribute `{attribute}` on `{}`",
                        self.context, function.name
                    ));
                }
                TestRegistration::Yes => {}
            }
            if let Some(reason) = &function.unclassifiable_macro {
                return Err(format!("{}: {reason} in `{}`", self.context, function.name));
            }
            functions.push(function);
        }
        Ok(functions)
    }

    fn parse_module_source(
        manifest_dir: &Path,
        module: &ModuleSource,
        source: &str,
        test_cfg: bool,
    ) -> Result<Self, String> {
        let relative = module
            .path
            .strip_prefix(manifest_dir)
            .unwrap_or(&module.path)
            .to_string_lossy()
            .into_owned();
        let mut parsed = Self::parse(&relative, source, test_cfg)?;
        parsed.external_cfg = module.external_cfg.clone();
        Ok(parsed)
    }
}

#[derive(Clone)]
struct StructuralGuard {
    protected: Range<usize>,
    binding: String,
    formula: CfgFormula,
    used_after_binding: bool,
}

#[derive(Clone, Copy)]
enum GuardRequirement {
    Lexical,
    Function { closing_brace: usize },
}

impl StructuredSource {
    fn statement_bounds(&self, site: usize) -> Option<(usize, usize, usize)> {
        let block = self.nearest_brace(site)?;
        let mut start = block + 1;
        for index in block + 1..site {
            let same_block =
                self.nearest_brace(index) == Some(block) || self.parents[index] == Some(block);
            if same_block && (self.tokens[index].is_punct(';') || self.tokens[index].is_punct('}'))
            {
                start = index + 1;
            }
        }
        let closing = self.pairs[block]?;
        let end = (site + 1..closing).find(|index| {
            self.tokens[*index].is_punct(';')
                && (self.nearest_brace(*index) == Some(block)
                    || self.parents[*index] == Some(block))
        })?;
        Some((start, end, block))
    }

    fn direct_guard_binding(
        &self,
        call: usize,
        selector: CallSelector,
    ) -> Result<Option<StructuralGuard>, String> {
        let Some((statement_start, statement_end, block)) = self.statement_bounds(call) else {
            return Ok(None);
        };
        let (_, content_start) = self.outer_attributes_at(statement_start);
        if self.tokens.get(content_start).and_then(RustToken::ident) != Some("let") {
            return Ok(None);
        }
        let Some(binding_token) = self.tokens.get(content_start + 1) else {
            return Ok(None);
        };
        let Some(binding) = binding_token.ident() else {
            return Ok(None);
        };
        if binding == "_" || binding_token.is_raw_ident() {
            return Ok(None);
        }
        if !self
            .tokens
            .get(content_start + 2)
            .is_some_and(|token| token.is_punct('='))
        {
            return Ok(None);
        }
        let call_start = match selector {
            CallSelector::Path(path) => self.path_start(call, path),
            CallSelector::Method(_) => Some(call),
        };
        if call_start != Some(content_start + 3) {
            return Ok(None);
        }
        let opening = call + 1;
        if !self
            .tokens
            .get(opening)
            .is_some_and(|token| token.is_punct('('))
        {
            return Ok(None);
        }
        let Some(closing) = self.pairs[opening] else {
            return Ok(None);
        };
        if closing != opening + 1 || statement_end != closing + 1 {
            return Ok(None);
        }
        let scope_end = self.pairs[block].ok_or_else(|| {
            format!(
                "{}: guard binding has no brace-balanced scope",
                self.context
            )
        })?;
        let used_after_binding = self.tokens[statement_end + 1..scope_end]
            .iter()
            .any(|token| token.ident() == Some(binding));
        Ok(Some(StructuralGuard {
            protected: statement_end + 1..scope_end,
            binding: binding.to_string(),
            formula: self.formula_at(call)?,
            used_after_binding,
        }))
    }

    fn guard_bindings(
        &self,
        range: Range<usize>,
        selector: CallSelector,
    ) -> Result<Vec<StructuralGuard>, String> {
        self.call_sites(range, selector, true)?
            .into_iter()
            .filter_map(|call| self.direct_guard_binding(call, selector).transpose())
            .collect()
    }

    fn validate_work_sites(
        &self,
        work_sites: &[usize],
        lock_selector: CallSelector,
        requirement: GuardRequirement,
    ) -> Result<(), String> {
        self.validate_work_sites_when(work_sites, lock_selector, requirement, &CfgFormula::True)
    }

    fn validate_work_sites_when(
        &self,
        work_sites: &[usize],
        lock_selector: CallSelector,
        requirement: GuardRequirement,
        active: &CfgFormula,
    ) -> Result<(), String> {
        if work_sites.is_empty() {
            return Err(format!("{}: protected work was not found", self.context));
        }
        let guards = self.guard_bindings(0..self.tokens.len(), lock_selector)?;
        let mut live_work = 0usize;
        for work in work_sites {
            let work_formula = CfgFormula::all([self.formula_at(*work)?, active.clone()]);
            if !work_formula.satisfiable()? {
                continue;
            }
            live_work += 1;
            let candidates = guards.iter().filter(|guard| {
                guard.protected.contains(work)
                    && !guard.used_after_binding
                    && match requirement {
                        GuardRequirement::Lexical => true,
                        GuardRequirement::Function { closing_brace } => {
                            guard.protected.end == closing_brace
                        }
                    }
            });
            let guard_formula = CfgFormula::any(candidates.map(|guard| guard.formula.clone()));
            if !work_formula.implies(&guard_formula)? {
                let moved = guards
                    .iter()
                    .find(|guard| guard.protected.contains(work) && guard.used_after_binding);
                let reason = moved.map_or_else(
                    || "no live shared-lock binding encloses this work".to_string(),
                    |guard| {
                        format!(
                            "shared-lock binding `{}` is used before its scope ends and may be moved, dropped, or shadowed",
                            guard.binding
                        )
                    },
                );
                return Err(format!(
                    "{}: {} at token byte {}",
                    self.context, reason, self.tokens[*work].offset
                ));
            }
        }
        if live_work == 0 {
            return Err(format!(
                "{}: protected work has no live occurrence in the Metal configuration",
                self.context
            ));
        }
        Ok(())
    }

    fn validate_selector(
        &self,
        range: Range<usize>,
        selector: CallSelector,
        lock_selector: CallSelector,
        requirement: GuardRequirement,
    ) -> Result<(), String> {
        let work_sites = self.call_sites(range, selector, true)?;
        self.validate_work_sites(&work_sites, lock_selector, requirement)
    }

    fn construction_sites(
        &self,
        range: Range<usize>,
    ) -> Result<Vec<(usize, CallSelector)>, String> {
        if let Some(reason) = &self.unclassifiable_test_scope {
            return Err(format!("{}: {reason}", self.context));
        }
        let mut sites = Vec::new();
        for selector in CONSTRUCTION_SELECTORS {
            for token in self.call_sites(range.clone(), *selector, true)? {
                sites.push((token, *selector));
            }
        }
        sites.sort_unstable_by_key(|(token, _)| *token);
        Ok(sites)
    }

    /// A key naming a construction site by the program structure enclosing
    /// it (`mod`/`impl`/`fn` path + constructed selector + an ordinal
    /// disambiguating repeated calls to the same selector in the same
    /// function) instead of by source line and column, so an edit that
    /// shifts line numbers without changing what is constructed or where it
    /// sits in the call graph does not invalidate an exemption keyed on it
    /// (#1357).
    fn stable_construction_key(
        &self,
        token: usize,
        selector: CallSelector,
        ordinal: usize,
    ) -> String {
        format!(
            "{}::{}::{}#{ordinal}",
            self.context,
            self.enclosing_function_path(token),
            selector.label()
        )
    }

    fn source_site_label(&self, token: usize) -> String {
        let offset = self.tokens[token].offset;
        let prefix = &self.source[..offset];
        let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let line_start = prefix.rfind('\n').map_or(0, |newline| newline + 1);
        let column = self.source[line_start..offset].chars().count() + 1;
        format!("{}:{line}:{column}", self.context)
    }

    fn has_top_level_metal_marker(&self) -> Result<bool, String> {
        if let Some(reason) = &self.unclassifiable_test_scope {
            return Err(format!("{}: {reason}", self.context));
        }
        for selector in RAW_GPU_SELECTORS {
            if self.has_selector(*selector)? {
                return Ok(true);
            }
        }
        Ok(["MetalQwen35State", "MetalForwardPass", "QwenModel"]
            .iter()
            .any(|marker| self.has_code_identifier(marker)))
    }
}

fn tokens_name_any(tokens: &proc_macro2::TokenStream, names: &BTreeSet<String>) -> bool {
    tokens.clone().into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(ident) => names.contains(&ident.to_string()),
        proc_macro2::TokenTree::Group(group) => tokens_name_any(&group.stream(), names),
        _ => false,
    })
}

struct BenchHarnessSyntax {
    names: BTreeSet<String>,
    call_paths: BTreeSet<String>,
    caller_scopes: BTreeSet<String>,
    item_path: Vec<String>,
    deferred: bool,
    function_depth: usize,
    error: Option<String>,
}

impl<'ast> Visit<'ast> for BenchHarnessSyntax {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func {
            let name = path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string());
            if name.is_some_and(|name| self.names.contains(&name))
                && (self.deferred || !self.call_paths.contains(&path_label(&path.path)))
            {
                self.error = Some("unreviewed or deferred call to a protected entrypoint".into());
            }
        } else {
            self.visit_expr(&call.func);
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path
            .path
            .segments
            .last()
            .is_some_and(|segment| self.names.contains(&segment.ident.to_string()))
        {
            self.error = Some("protected entrypoint used as a value or alias".into());
        }
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_closure(self, closure);
        self.deferred = previous;
    }

    fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
        let previous = self.deferred;
        self.deferred = true;
        syn::visit::visit_expr_async(self, expression);
        self.deferred = previous;
    }

    fn visit_item_fn(&mut self, function: &'ast ItemFn) {
        if function.sig.asyncness.is_some() && self.names.contains(&function.sig.ident.to_string())
        {
            self.error = Some("protected entrypoint must execute synchronously".into());
        }
        if self.function_depth > 0 && self.names.contains(&function.sig.ident.to_string()) {
            self.error = Some("protected entrypoint shadowed by a local function".into());
        }
        let previous = self.deferred;
        self.deferred |= self.function_depth > 0;
        self.function_depth += 1;
        self.item_path.push(function.sig.ident.to_string());
        syn::visit::visit_item_fn(self, function);
        self.item_path.pop();
        self.function_depth -= 1;
        self.deferred = previous;
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        self.item_path.push(module.ident.to_string());
        syn::visit::visit_item_mod(self, module);
        self.item_path.pop();
    }

    fn visit_use_tree(&mut self, tree: &'ast syn::UseTree) {
        match tree {
            syn::UseTree::Name(name) if self.names.contains(&name.ident.to_string()) => {
                self.error =
                    Some("protected entrypoint imported through an unsupported path".into());
            }
            syn::UseTree::Rename(rename)
                if self.names.contains(&rename.ident.to_string())
                    || self.names.contains(&rename.rename.to_string()) =>
            {
                self.error = Some("protected entrypoint imported through an alias".into());
            }
            syn::UseTree::Glob(_)
                if self.function_depth == 0
                    || self.caller_scopes.contains(&self.item_path.join("::")) =>
            {
                self.error = Some("glob import can shadow a reviewed caller edge".into())
            }
            _ => syn::visit::visit_use_tree(self, tree),
        }
    }

    fn visit_pat_ident(&mut self, binding: &'ast syn::PatIdent) {
        if self.names.contains(&binding.ident.to_string()) {
            self.error = Some("protected entrypoint shadowed by a binding".into());
        }
        syn::visit::visit_pat_ident(self, binding);
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        if self.names.contains(&item.ident.to_string()) {
            self.error = Some("protected entrypoint shadowed by a constant".into());
        }
        syn::visit::visit_item_const(self, item);
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        if self.names.contains(&item.ident.to_string()) {
            self.error = Some("protected entrypoint shadowed by a static".into());
        }
        syn::visit::visit_item_static(self, item);
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        if self.names.contains(&item.ident.to_string()) {
            self.error = Some("protected entrypoint shadowed by a struct constructor".into());
        }
        syn::visit::visit_item_struct(self, item);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if !mac.path.is_ident("criterion_group")
            && !mac.path.is_ident("criterion_main")
            && tokens_name_any(&mac.tokens, &self.names)
        {
            self.error = Some("protected entrypoint in an unsupported macro".into());
        }
    }
}

fn validate_bench_wrapper_span(
    parsed: &StructuredSource,
    syntax: &syn::File,
    contract: &BenchHarness,
    active: &CfgFormula,
) -> Result<(), String> {
    if contract.registered.len() != 1 {
        return Err("benchmark target list must have one registered wrapper".into());
    }
    let expected = contract
        .calls
        .iter()
        .filter(|call| call.guarded)
        .map(|call| call.selector.label().trim_end_matches("()").to_string())
        .collect::<Vec<_>>();
    let mut functions = Vec::new();
    module_functions(&syntax.items, &mut functions);
    let mut active_wrappers = 0;
    for function in functions {
        let body =
            parsed.token_range_for_block(&function.block, &function.sig.ident.to_string())?;
        if parsed.enclosing_function_path(body.start) != contract.registered[0] {
            continue;
        }
        let formula = CfgFormula::all([parsed.formula_at(body.start)?, active.clone()]);
        if !formula.satisfiable()? {
            continue;
        }
        if !parsed
            .functions
            .iter()
            .any(|candidate| candidate.body == body && candidate.returns_unit)
        {
            return Err("registered benchmark wrapper must return unit synchronously".into());
        }
        active_wrappers += 1;
        let parameters = function
            .sig
            .inputs
            .iter()
            .map(|input| match input {
                syn::FnArg::Typed(input) => match &*input.pat {
                    syn::Pat::Ident(binding)
                        if binding.by_ref.is_none() && binding.subpat.is_none() =>
                    {
                        Ok(binding.ident.to_string())
                    }
                    _ => Err("wrapper parameters must be plain named bindings".to_string()),
                },
                _ => Err("wrapper parameters must be plain named bindings".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut actual = Vec::new();
        let mut sites = Vec::new();
        let mut locals = 0;
        for statement in &function.block.stmts {
            match statement {
                syn::Stmt::Local(_) => locals += 1,
                syn::Stmt::Expr(syn::Expr::Call(call), _) => {
                    if !call.attrs.is_empty() {
                        return Err(
                            "wrapper arms must not have conditional or other call attributes"
                                .into(),
                        );
                    }
                    if call.args.len() != parameters.len()
                        || call
                            .args
                            .iter()
                            .zip(&parameters)
                            .any(|(argument, parameter)| {
                                !matches!(argument, syn::Expr::Path(path)
                            if path.attrs.is_empty() && path.qself.is_none()
                                && path.path.is_ident(parameter.as_str()))
                            })
                    {
                        return Err(
                            "wrapper arms must pass only unchanged bare wrapper parameters".into(),
                        );
                    }
                    let syn::Expr::Path(path) = &*call.func else {
                        return Err("wrapper arms must be immediate named calls".into());
                    };
                    actual.push(path_label(&path.path));
                    let offset = path
                        .path
                        .segments
                        .last()
                        .ok_or("empty wrapper arm path")?
                        .ident
                        .span()
                        .byte_range()
                        .start;
                    sites.push(
                        parsed
                            .tokens
                            .iter()
                            .position(|token| token.offset == offset)
                            .ok_or("wrapper arm could not be mapped to parsed source")?,
                    );
                }
                _ => {
                    return Err(
                        "wrapper must contain only one guard binding and immediate reviewed arms"
                            .into(),
                    );
                }
            }
        }
        if actual != expected {
            return Err(format!(
                "wrapper arm order changed: expected {expected:?}, found {actual:?}"
            ));
        }
        let guards = parsed.guard_bindings(body.clone(), SHARED_LOCK_SELECTOR)?;
        if locals != 1 || guards.len() != 1 {
            return Err(
                "one shared-lock binding must span the complete benchmark target list".into(),
            );
        }
        let guard = &guards[0];
        if guard.used_after_binding
            || guard.protected.end != body.end
            || sites.iter().any(|site| !guard.protected.contains(site))
            || !formula.implies(&guard.formula)?
        {
            return Err("the same live guard must cover every arm through wrapper return".into());
        }
    }
    if active_wrappers == 0 {
        return Err("registered wrapper has no active Metal definition".into());
    }
    Ok(())
}

fn validate_bench_harness(
    parsed: &StructuredSource,
    contract: &BenchHarness,
) -> Result<(), String> {
    let syntax = syn::parse_file(&parsed.source).map_err(|reason| reason.to_string())?;
    let mut macros = MacroCollector { macros: Vec::new() };
    macros.visit_file(&syntax);
    let registration = |name: &str| -> Result<Vec<String>, String> {
        let matching = macros
            .macros
            .iter()
            .filter(|mac| mac.path.is_ident(name))
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!("expected one {name}! registration"));
        }
        if !syntax.items.iter().any(|item| {
            matches!(item, syn::Item::Macro(item)
            if item.mac.path.is_ident(name) && item.attrs.is_empty())
        }) {
            return Err(format!(
                "{name}! must be an unconditional root registration"
            ));
        }
        let arguments = Punctuated::<syn::Ident, Token![,]>::parse_terminated
            .parse2(matching[0].tokens.clone())
            .map_err(|reason| format!("unsupported {name}! registration: {reason}"))?;
        Ok(arguments.iter().map(ToString::to_string).collect())
    };
    let group = registration("criterion_group")?;
    let main = registration("criterion_main")?;
    if group.is_empty() || main != group[..1] || group[1..] != *contract.registered {
        return Err(
            "Criterion must register exactly the reviewed guard owners through its main group"
                .into(),
        );
    }
    let mut shape = BenchHarnessSyntax {
        names: contract
            .registered
            .iter()
            .map(|name| (*name).to_string())
            .chain(
                contract
                    .calls
                    .iter()
                    .map(|call| call.selector.final_name().to_string()),
            )
            .collect(),
        call_paths: contract
            .calls
            .iter()
            .map(|call| call.selector.label().trim_end_matches("()").to_string())
            .collect(),
        caller_scopes: contract
            .registered
            .iter()
            .copied()
            .chain(
                contract
                    .calls
                    .iter()
                    .flat_map(|call| call.callers.iter().copied()),
            )
            .map(str::to_string)
            .collect(),
        item_path: Vec::new(),
        deferred: false,
        function_depth: 0,
        error: None,
    };
    shape.visit_file(&syntax);
    if let Some(error) = shape.error {
        return Err(error);
    }

    let constructions = parsed.construction_sites(0..parsed.tokens.len())?;
    let mut actual = constructions
        .iter()
        .map(|(site, selector)| (parsed.enclosing_function_path(*site), selector.label()))
        .collect::<Vec<_>>();
    let mut expected = contract
        .constructions
        .iter()
        .map(|(owner, selector)| ((*owner).to_string(), selector.label()))
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    if actual != expected {
        return Err(format!(
            "state construction ownership changed: expected {expected:?}, found {actual:?}"
        ));
    }
    let mut active_sites = constructions
        .iter()
        .map(|(site, _)| *site)
        .collect::<Vec<_>>();
    for selector in RAW_GPU_SELECTORS {
        active_sites.extend(parsed.call_sites(0..parsed.tokens.len(), *selector, true)?);
    }
    let active = CfgFormula::any(
        active_sites
            .iter()
            .map(|site| parsed.formula_at(*site))
            .collect::<Result<Vec<_>, _>>()?,
    );
    if !active.satisfiable()? {
        return Err("benchmark harness has no active Metal work".into());
    }
    for call in contract.calls {
        if call.guarded {
            let callee = call.selector.label().trim_end_matches("()").to_string();
            let owners = parsed
                .functions
                .iter()
                .filter(|function| parsed.enclosing_function_path(function.body.start) == callee)
                .collect::<Vec<_>>();
            if owners.is_empty() || owners.iter().any(|owner| !owner.returns_unit) {
                return Err(format!(
                    "guarded measurement {callee} must complete without returning a live handle"
                ));
            }
        }
        let sites = parsed.call_sites(0..parsed.tokens.len(), call.selector, true)?;
        let mut callers = BTreeSet::new();
        for site in sites {
            if !CfgFormula::all([parsed.formula_at(site)?, active.clone()]).satisfiable()? {
                continue;
            }
            let caller = parsed.enclosing_function_path(site);
            if !call.callers.contains(&caller.as_str()) {
                return Err(format!(
                    "{} has unreviewed caller {caller}",
                    call.selector.label()
                ));
            }
            callers.insert(caller);
            if call.guarded {
                parsed.validate_work_sites_when(
                    &[site],
                    SHARED_LOCK_SELECTOR,
                    GuardRequirement::Lexical,
                    &active,
                )?;
            }
        }
        if callers
            != call
                .callers
                .iter()
                .map(|caller| (*caller).to_string())
                .collect()
        {
            return Err(format!(
                "{} does not have exactly its reviewed active callers",
                call.selector.label()
            ));
        }
    }
    if !contract.calls.is_empty() {
        for site in parsed.call_sites(0..parsed.tokens.len(), SHARED_LOCK_SELECTOR, true)? {
            if !contract
                .registered
                .contains(&parsed.enclosing_function_path(site).as_str())
            {
                return Err("shared lock must be owned by the registered wrapper, not reacquired by a callee".into());
            }
        }
        validate_bench_wrapper_span(parsed, &syntax, contract, &active)?;
    }
    if contract.calls.is_empty() {
        let mut functions = Vec::new();
        module_functions(&syntax.items, &mut functions);
        for (site, _) in constructions {
            let owner = parsed.enclosing_function_path(site);
            let body = &parsed
                .function_paths
                .iter()
                .find(|(_, path)| *path == owner)
                .ok_or_else(|| format!("missing measurement owner {owner}"))?
                .0;
            let function = functions
                .iter()
                .find(|function| {
                    parsed
                        .token_range_for_block(&function.block, &owner)
                        .as_ref()
                        == Ok(body)
                })
                .ok_or_else(|| format!("missing registered function {owner}"))?;
            // A cfg-dispatched owner may consist entirely of alternate blocks.
            // Require one complete active block, not a constructor's short scope.
            let blocks = function
                .block
                .stmts
                .iter()
                .map(|statement| match statement {
                    syn::Stmt::Expr(syn::Expr::Block(block), _) if !block.attrs.is_empty() => {
                        Some(block)
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            let end = if let Some(blocks) = blocks {
                let mut active_blocks = Vec::new();
                for block in blocks {
                    let formula =
                        syn_attributes_formula(&block.attrs, parsed.test_cfg, &parsed.context)?;
                    if CfgFormula::all([active.clone(), formula]).satisfiable()? {
                        active_blocks.push(parsed.token_range_for_block(&block.block, &owner)?);
                    }
                }
                if active_blocks.len() != 1 || !active_blocks[0].contains(&site) {
                    return Err(format!(
                        "registered owner {owner} must have one complete active measurement block"
                    ));
                }
                active_blocks[0].end
            } else {
                body.end
            };
            parsed.validate_work_sites(
                &[site],
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Function { closing_brace: end },
            )?;
        }
    }
    Ok(())
}

fn validate_bench_harness_sources(
    sources: &[ParsedModuleSource],
    contract: &BenchHarness,
) -> Result<(), String> {
    if sources.len() != 1 || sources[0].relative != contract.path {
        return Err(
            "reviewed state harness module closure changed; classify its new source paths".into(),
        );
    }
    validate_bench_harness(&sources[0].parsed, contract)
}

fn validate_bench_harness_inventory(
    reviewed: &BTreeSet<String>,
    covered: &BTreeSet<String>,
) -> Result<(), String> {
    if reviewed == covered {
        Ok(())
    } else {
        Err("reviewed Metal benchmark lifetime coverage changed".into())
    }
}

#[test]
fn raw_metal_measurement_harnesses_use_live_lock_bindings_across_the_entrypoint() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut actual = BTreeSet::new();
    let reviewed = TARGETS_WITH_RECOGNIZED_METAL_MARKERS
        .iter()
        .filter(|path| path.starts_with("benches/"))
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    let covered = BENCH_HARNESSES
        .iter()
        .map(|contract| contract.path)
        .map(str::to_string)
        .collect();
    validate_bench_harness_inventory(&reviewed, &covered)
        .unwrap_or_else(|reason| panic!("{reason}"));
    assert!(
        BENCH_HARNESSES
            .iter()
            .all(|contract| !contract.calls.is_empty() && contract.registered.len() == 1),
        "every reviewed Metal benchmark needs one complete target-list wrapper"
    );
    assert_eq!(
        BENCH_HARNESSES
            .iter()
            .map(|contract| contract.path)
            .collect::<BTreeSet<_>>()
            .len(),
        BENCH_HARNESSES.len(),
        "state harness ownership records must not repeat paths"
    );
    let mut state_actual = BTreeSet::new();
    let mut violations = Vec::new();

    for path in cargo_target_roots(manifest_dir, &["bench", "example", "bin"])
        .unwrap_or_else(|reason| panic!("{reason}"))
    {
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        let sources = parsed_module_closure(manifest_dir, &path, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        if let Some(contract) = BENCH_HARNESSES
            .iter()
            .find(|contract| contract.path == relative)
        {
            state_actual.insert(relative.clone());
            if let Err(reason) = validate_bench_harness_sources(&sources, contract) {
                violations.push(format!("{relative}: {reason}"));
            }
        }
        let mut has_raw_gpu_work = false;
        for source in &sources {
            for selector in RAW_GPU_SELECTORS {
                has_raw_gpu_work |= source
                    .parsed
                    .has_selector(*selector)
                    .unwrap_or_else(|reason| panic!("{reason}"));
            }
        }
        if !has_raw_gpu_work {
            continue;
        }
        actual.insert(relative.clone());

        let protected_entrypoint = RAW_HARNESS_ENTRYPOINTS
            .iter()
            .find_map(|(path, marker)| (*path == relative).then_some(*marker))
            .unwrap_or_else(|| panic!("raw Metal harness {relative} is not classified"));
        let root = sources
            .iter()
            .find(|source| source.relative == relative)
            .unwrap_or_else(|| panic!("target root {relative} was not parsed"));
        root.parsed
            .validate_selector(
                0..root.parsed.tokens.len(),
                protected_entrypoint,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
            .unwrap_or_else(|reason| panic!("{reason}"));
    }

    let expected = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .map(|(path, _)| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "raw Metal harness inventory changed; classify every added or removed path explicitly"
    );
    assert_eq!(
        state_actual,
        BENCH_HARNESSES
            .iter()
            .map(|contract| contract.path.to_string())
            .collect(),
        "state harness ownership records must name current Cargo targets"
    );
    assert!(
        violations.is_empty(),
        "state Metal measurement ownership violations:\n{}",
        violations.join("\n")
    );
}

const STATE_HARNESS_FIXTURE: BenchHarness = BenchHarness {
    path: "fixtures/state_harness.rs",
    registered: &["bench_locked"],
    calls: &[
        BenchHarnessCall {
            selector: CallSelector::Path(&["measure"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["load"]),
            callers: &["measure"],
            guarded: false,
        },
    ],
    constructions: &[("load", CallSelector::Path(&["MetalQwen35State", "new"]))],
};

const STATE_HARNESS_SOURCE: &str = r#"
fn load() -> MetalQwen35State { MetalQwen35State::new() }
fn measure() { let mut state = load(); state.forward_step(); }
fn bench_locked() {
    let guard = lattice_inference::measurement::gpu_test_lock();
    measure();
}
criterion_group!(benches, bench_locked);
criterion_main!(benches);
"#;

fn state_harness_fixture(source: &str, contract: &BenchHarness) -> Result<(), String> {
    let parsed = StructuredSource::parse(contract.path, source, false)?;
    validate_bench_harness(&parsed, contract)
}

#[test]
fn state_harness_rejects_setup_only_guard_with_registered_wrapper_intact() {
    state_harness_fixture(STATE_HARNESS_SOURCE, &STATE_HARNESS_FIXTURE)
        .expect("caller owns returned state lifetime");
    let source = STATE_HARNESS_SOURCE
        .replace("    let guard = lattice_inference::measurement::gpu_test_lock();\n", "")
        .replace("fn load() -> MetalQwen35State {", "fn load() -> MetalQwen35State { let guard = lattice_inference::measurement::gpu_test_lock();");
    let error = state_harness_fixture(&source, &STATE_HARNESS_FIXTURE)
        .expect_err("setup-only guard must fail");
    assert!(
        error.contains("no live shared-lock binding"),
        "wrong failure: {error}"
    );
}

#[test]
fn state_harness_rejects_unreviewed_callers_aliases_and_deferred_work() {
    for replacement in [
        "fn extra() { let state = load(); }",
        "fn extra() { let alias = load; alias(); }",
        "use other::function as load;",
        "fn extra(load: fn()) { load(); }",
    ] {
        let source = format!("{STATE_HARNESS_SOURCE}\n{replacement}");
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {replacement}"
        );
    }
    for replacement in [
        "let delayed = || measure(); delayed();",
        "let future = async { measure(); };",
        "let measure = other; measure();",
        "fn measure() {} measure();",
        "const measure: fn() = other; measure();",
        "static measure: fn() = other; measure();",
        "struct measure(); measure();",
    ] {
        let source = STATE_HARNESS_SOURCE.replace("    measure();", replacement);
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {replacement}"
        );
    }
    let asynchronous = STATE_HARNESS_SOURCE.replace("fn measure()", "async fn measure()");
    assert!(state_harness_fixture(&asynchronous, &STATE_HARNESS_FIXTURE).is_err());
}

#[test]
fn state_harness_rejects_extra_or_inactive_criterion_registration() {
    for registration in [
        "criterion_group!(benches, bench_locked, measure);",
        "#[cfg(any())] criterion_group!(benches, bench_locked);",
        "mod inactive { criterion_group!(benches, bench_locked); }",
        "criterion_group!(other, bench_locked);",
    ] {
        let source =
            STATE_HARNESS_SOURCE.replace("criterion_group!(benches, bench_locked);", registration);
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {registration}"
        );
    }
}

#[test]
fn state_harness_rejects_guard_decoys_and_nested_reacquisition() {
    for guard in [
        "#[cfg(any())] let guard = lattice_inference::measurement::gpu_test_lock();",
        "let guard = lattice_inference::measurement::gpu_test_lock(); drop(guard);",
        "{ let guard = lattice_inference::measurement::gpu_test_lock(); }",
    ] {
        let source = STATE_HARNESS_SOURCE.replace(
            "let guard = lattice_inference::measurement::gpu_test_lock();",
            guard,
        );
        assert!(
            state_harness_fixture(&source, &STATE_HARNESS_FIXTURE).is_err(),
            "accepted {guard}"
        );
    }
    let nested = STATE_HARNESS_SOURCE.replace("fn load() -> MetalQwen35State {", "fn load() -> MetalQwen35State { let guard = lattice_inference::measurement::gpu_test_lock();");
    let error = state_harness_fixture(&nested, &STATE_HARNESS_FIXTURE)
        .expect_err("nested acquisition must fail");
    assert!(
        error.contains("reacquired by a callee"),
        "wrong failure: {error}"
    );
}

#[test]
fn state_harness_accepts_completed_results_but_rejects_setup_scope_escape() {
    let contract = BenchHarness {
        path: "fixtures/complete_operation.rs",
        registered: &["run"],
        calls: &[],
        constructions: &[("run", CallSelector::Path(&["MetalQwen35State", "new"]))],
    };
    for output in ["", " -> Vec<Metric>"] {
        let tail = if output.is_empty() { "" } else { "vec![]" };
        let source = format!(
            r#"
fn run(){output} {{
    let guard = lattice_inference::measurement::gpu_test_lock();
    let mut state = MetalQwen35State::new();
    state.forward_step();
    {tail}
}}
criterion_group!(benches, run);
criterion_main!(benches);
"#
        );
        state_harness_fixture(&source, &contract)
            .expect("complete operation may return measurements");
    }
    let source = r#"
fn run() {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let mut state;
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))] {
        let guard = lattice_inference::measurement::gpu_test_lock();
        state = MetalQwen35State::new();
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))] {
        state.chat_completion();
    }
}
criterion_group!(benches, run);
criterion_main!(benches);
"#;
    let error = state_harness_fixture(source, &contract).expect_err("state escaped setup guard");
    assert!(
        error.contains("no live shared-lock binding"),
        "wrong failure: {error}"
    );
    let complete_block = source.replace(
        "    }\n    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))] {\n        state.chat_completion();",
        "        state.chat_completion();",
    ).replace("    #[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]\n    let mut state;\n", "")
        .replace("state = MetalQwen35State::new();", "let mut state = MetalQwen35State::new();");
    state_harness_fixture(&complete_block, &contract)
        .expect("whole active block owns measurement lifetime");
}

#[test]
fn state_harness_inventory_rejects_omitted_stale_and_new_paths() {
    let reviewed = BTreeSet::from(["benches/known.rs".to_string()]);
    validate_bench_harness_inventory(&reviewed, &reviewed).expect("exact coverage");
    for covered in [
        BTreeSet::new(),
        BTreeSet::from(["benches/stale.rs".to_string()]),
        BTreeSet::from(["benches/known.rs".to_string(), "benches/new.rs".to_string()]),
    ] {
        assert!(validate_bench_harness_inventory(&reviewed, &covered).is_err());
    }
}

const MULTI_ARM_HARNESS_FIXTURE: BenchHarness = BenchHarness {
    path: "fixtures/whole_target_list.rs",
    registered: &["bench_locked"],
    calls: &[
        BenchHarnessCall {
            selector: CallSelector::Path(&["measure"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["cpu"]),
            callers: &["bench_locked"],
            guarded: true,
        },
        BenchHarnessCall {
            selector: CallSelector::Path(&["load"]),
            callers: &["measure"],
            guarded: false,
        },
    ],
    constructions: &[("load", CallSelector::Path(&["MetalQwen35State", "new"]))],
};

const COMPLETE_TARGET_LIST: &str =
    "let guard = lattice_inference::measurement::gpu_test_lock(); measure(); cpu();";

fn multi_arm_harness_source(body: &str) -> String {
    format!(
        r#"
fn load() -> MetalQwen35State {{ MetalQwen35State::new() }}
fn measure() {{ let mut state = load(); state.forward_step(); }}
fn cpu() {{ work(); }}
fn bench_locked() {{ {body} }}
criterion_group!(benches, bench_locked);
criterion_main!(benches);
"#
    )
}

#[test]
fn benchmark_target_list_requires_one_guard_for_every_ordered_arm() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    state_harness_fixture(&source, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("one span covers GPU and CPU arms");
    let separate_guards = multi_arm_harness_source(
        "{ let first = lattice_inference::measurement::gpu_test_lock(); measure(); }
         { let second = lattice_inference::measurement::gpu_test_lock(); cpu(); }",
    );
    let error = state_harness_fixture(&separate_guards, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("individual arm guards cannot certify one continuous span");
    assert!(
        error.contains("immediate reviewed arms"),
        "wrong failure: {error}"
    );
    for arms in [
        "cpu(); measure();",
        "measure();",
        "measure(); cpu(); cpu();",
        "measure(); #[cfg(feature = \"extra\")] cpu();",
    ] {
        let body = format!("let guard = lattice_inference::measurement::gpu_test_lock(); {arms}");
        assert!(
            state_harness_fixture(&multi_arm_harness_source(&body), &MULTI_ARM_HARNESS_FIXTURE)
                .is_err(),
            "accepted {arms}"
        );
    }
    let cpu_outside = multi_arm_harness_source(
        "{ let guard = lattice_inference::measurement::gpu_test_lock(); measure(); } cpu();",
    );
    assert!(state_harness_fixture(&cpu_outside, &MULTI_ARM_HARNESS_FIXTURE).is_err());
    let registered_sibling = source.replace(
        "criterion_group!(benches, bench_locked);",
        "criterion_group!(benches, bench_locked, cpu);",
    );
    let error = state_harness_fixture(&registered_sibling, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("unprotected CPU sibling registration");
    assert!(
        error.contains("Criterion must register exactly"),
        "wrong failure: {error}"
    );
    let asynchronous = source.replace("fn bench_locked()", "async fn bench_locked()");
    assert!(state_harness_fixture(&asynchronous, &MULTI_ARM_HARNESS_FIXTURE).is_err());
}

#[test]
fn benchmark_target_list_checks_every_active_wrapper_definition() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    let wrapper = format!("fn bench_locked() {{ {COMPLETE_TARGET_LIST} }}");
    let variants = |other: &str| {
        format!(
            r#"
#[cfg(feature = "variant")]
fn bench_locked() {{ {COMPLETE_TARGET_LIST} }}
#[cfg(not(feature = "variant"))]
fn bench_locked() {{ {other} }}
"#
        )
    };
    let valid = source.replace(&wrapper, &variants(COMPLETE_TARGET_LIST));
    state_harness_fixture(&valid, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("both active alternatives have complete spans");
    let invalid = source.replace(
        &wrapper,
        &variants(
            "{ let first = lattice_inference::measurement::gpu_test_lock(); measure(); }
         { let second = lattice_inference::measurement::gpu_test_lock(); cpu(); }",
        ),
    );
    assert!(
        state_harness_fixture(&invalid, &MULTI_ARM_HARNESS_FIXTURE).is_err(),
        "one correct cfg variant cannot certify the other"
    );
}

#[test]
fn benchmark_leaf_globs_cannot_shadow_reviewed_caller_edges() {
    let source = multi_arm_harness_source(COMPLETE_TARGET_LIST);
    let leaf = source.replace("fn cpu() {", "fn cpu() { use metal::*;");
    state_harness_fixture(&leaf, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("leaf import does not affect the wrapper's lexical scope");
    for declaration in ["fn bench_locked() {", "fn measure() {"] {
        let changed = source.replace(declaration, &format!("{declaration} use metal::*;"));
        let error = state_harness_fixture(&changed, &MULTI_ARM_HARNESS_FIXTURE)
            .expect_err("glob in caller scope");
        assert!(
            error.contains("glob import can shadow"),
            "wrong failure: {error}"
        );
    }
    assert!(
        state_harness_fixture(
            &format!("use metal::*;\n{source}"),
            &MULTI_ARM_HARNESS_FIXTURE
        )
        .is_err()
    );
    let extra = format!("{source}\nfn extra() {{ use metal::*; load(); }}");
    let error = state_harness_fixture(&extra, &MULTI_ARM_HARNESS_FIXTURE)
        .expect_err("glob does not admit an extra caller");
    assert!(
        error.contains("unreviewed caller extra"),
        "wrong failure: {error}"
    );
}

#[test]
fn benchmark_arm_arguments_cannot_hide_control_flow_or_replace_parameters() {
    let source = multi_arm_harness_source(
        "let guard = lattice_inference::measurement::gpu_test_lock(); measure(c); cpu(c);",
    )
    .replace("fn bench_locked()", "fn bench_locked(c: &mut Criterion)")
    .replace("fn measure()", "fn measure(c: &mut Criterion)")
    .replace("fn cpu()", "fn cpu(c: &mut Criterion)");
    state_harness_fixture(&source, &MULTI_ARM_HARNESS_FIXTURE)
        .expect("the original Criterion parameter reaches both arms unchanged");
    for argument in ["{ return; }", "other", "&mut *c"] {
        let changed = source.replace("cpu(c);", &format!("cpu({argument});"));
        let error = state_harness_fixture(&changed, &MULTI_ARM_HARNESS_FIXTURE)
            .expect_err("arm argument must not change control flow or the passed value");
        assert!(
            error.contains("unchanged bare wrapper parameters"),
            "wrong failure: {error}"
        );
    }
}

/// Matches every Cargo-selected executable target against recognized source markers.
///
/// This is a pre-expansion lexical inventory. It does not classify work whose
/// protected identifiers appear only after macro expansion or name resolution.
#[test]
fn cargo_target_lexical_metal_marker_inventory_is_explicit() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut discovered = BTreeSet::new();
    let mut all_targets = BTreeSet::new();
    for path in cargo_target_roots(manifest_dir, &["bench", "example", "bin"])
        .unwrap_or_else(|reason| panic!("{reason}"))
    {
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        all_targets.insert(relative.clone());
        let sources = parsed_module_closure(manifest_dir, &path, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        if sources.iter().any(|source| {
            source
                .parsed
                .has_top_level_metal_marker()
                .unwrap_or_else(|reason| panic!("{reason}"))
        }) {
            discovered.insert(relative);
        }
    }

    let reviewed = TARGETS_WITH_RECOGNIZED_METAL_MARKERS
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "top-level Metal entrypoint inventory changed; classify direct, helper-mediated, and \
         alternate-state-family paths explicitly"
    );
    let reviewed_non_metal = TARGETS_WITHOUT_RECOGNIZED_METAL_MARKERS
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert!(
        reviewed.is_disjoint(&reviewed_non_metal),
        "top-level targets must have exactly one classification"
    );
    let all_reviewed = reviewed
        .union(&reviewed_non_metal)
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        all_targets, all_reviewed,
        "top-level target inventory changed; every new bench, example, or binary must be \
         classified before it can pass the contract"
    );
}

#[test]
fn alternate_and_helper_mediated_entrypoints_hold_the_shared_lock() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (relative, entrypoint) in ADDITIONAL_GUARDED_ENTRYPOINTS {
        let source = std::fs::read_to_string(manifest_dir.join(relative))
            .expect("read guarded Metal entrypoint");
        let parsed = StructuredSource::parse(relative, &source, false)
            .unwrap_or_else(|reason| panic!("{reason}"));
        parsed
            .validate_selector(
                0..parsed.tokens.len(),
                *entrypoint,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
            .unwrap_or_else(|reason| panic!("{reason}"));
    }
}

#[test]
fn in_crate_raw_command_buffer_tests_are_explicit_and_guarded() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut discovered = BTreeSet::new();
    let mut violations = Vec::new();
    for root in cargo_target_roots(manifest_dir, &["lib", "test"])
        .unwrap_or_else(|reason| panic!("{reason}"))
    {
        for source in parsed_module_closure(manifest_dir, &root, true)
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            let relative = source.relative;
            let parsed = source.parsed;
            let functions = parsed
                .test_functions()
                .unwrap_or_else(|reason| panic!("{reason}"));
            for selector in COMMAND_BUFFER_SELECTORS {
                parsed
                    .reject_selector_in_macros(*selector)
                    .unwrap_or_else(|reason| panic!("{reason}"));
            }
            for function in functions {
                let mut command_buffers = Vec::new();
                for selector in COMMAND_BUFFER_SELECTORS {
                    command_buffers.extend(
                        parsed
                            .call_sites(function.body.clone(), *selector, true)
                            .unwrap_or_else(|reason| panic!("{reason}")),
                    );
                }
                if command_buffers.is_empty() {
                    continue;
                }
                let site = format!("{relative}::{}", function.name);
                discovered.insert(site.clone());
                if let Err(reason) = parsed.validate_work_sites(
                    &command_buffers,
                    LOCAL_LOCK_SELECTOR,
                    GuardRequirement::Function {
                        closing_brace: function.body.end,
                    },
                ) {
                    violations.push(format!("{site}: {reason}"));
                }
            }
        }
    }

    let reviewed = IN_CRATE_COMMAND_BUFFER_TESTS
        .iter()
        .map(|site| (*site).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "in-crate raw command-buffer inventory changed; classify every test explicitly"
    );
    assert!(
        violations.is_empty(),
        "in-crate raw command-buffer tests without a function-lifetime shared lock:\n{}",
        violations.join("\n")
    );
}

fn assert_helper_mediated_test_holds_function_lifetime_lock(function_name: &str) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let relative = "src/forward/metal_qwen35.rs";
    let source = std::fs::read_to_string(manifest_dir.join(relative))
        .expect("read helper-mediated Metal tests");
    let parsed = StructuredSource::parse(relative, &source, true)
        .unwrap_or_else(|reason| panic!("{reason}"));
    let function = parsed
        .test_functions()
        .unwrap_or_else(|reason| panic!("{reason}"))
        .into_iter()
        .find(|function| function.name == function_name)
        .unwrap_or_else(|| panic!("checked helper-mediated test `{function_name}` was not found"));
    parsed
        .validate_selector(
            function.body.clone(),
            DEVICE_SYSTEM_DEFAULT,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .unwrap_or_else(|reason| panic!("{relative}::{function_name}: {reason}"));
}

#[test]
fn mtp_draft_logit_equivalence_test_holds_a_function_lifetime_lock() {
    assert_helper_mediated_test_holds_function_lifetime_lock(
        "mtp_draft_logit_equivalence_with_quarot_counter_rotation",
    );
}

#[test]
fn metal_engine_session_isolation_test_holds_a_function_lifetime_lock() {
    assert_helper_mediated_test_holds_function_lifetime_lock(
        "test_metal_qwen35_engine_session_isolation",
    );
}

/// Records the pre-expansion boundary of the checked binding convention.
///
/// Macro metavariables can expand into protected Metal work after this scanner
/// has run, so this source-level check intentionally does not classify them.
#[test]
fn macro_metavariable_expansion_is_outside_the_checked_binding_convention() {
    let source = r#"
macro_rules! submit {
    ($metal:ident, $device:ident, $system:ident, $queue:ident, $buffer:ident) => {{
        let device = $metal::$device::$system().unwrap();
        let queue = device.$queue();
        let command_buffer = queue.$buffer();
        command_buffer.commit();
    }};
}

fn main() {
    submit!(metal, Device, system_default, new_command_queue, new_command_buffer);
}
"#;
    let parsed = StructuredSource::parse("fixtures/macro_metavariable_limit.rs", source, false)
        .expect("parse macro-metavariable limit fixture");
    assert!(
        !parsed
            .has_top_level_metal_marker()
            .expect("classify the documented macro-metavariable limit"),
        "the pre-expansion convention unexpectedly claimed macro-expanded work"
    );
}

#[test]
fn malformed_rust_fails_closed_in_direct_source_parsing() {
    let result = StructuredSource::parse("fixtures/malformed_direct.rs", "fn broken( {", false);
    let Err(message) = result else {
        panic!("direct parser accepted malformed Rust");
    };
    assert!(
        message.contains("Rust syntax could not be classified"),
        "direct parser reported the wrong failure: {message}"
    );
}

#[test]
fn malformed_rust_fails_closed_in_module_closure_parsing() {
    let fixture = tempfile::tempdir().expect("create malformed module fixture");
    let root = fixture.path().join("main.rs");
    std::fs::write(&root, "mod broken;").expect("write fixture root");
    std::fs::write(fixture.path().join("broken.rs"), "fn broken( {")
        .expect("write malformed module");

    let result = module_source_closure(fixture.path(), &root, false);
    let Err(message) = result else {
        panic!("module closure accepted malformed Rust");
    };
    assert!(
        message.contains("broken.rs: Rust syntax could not be classified"),
        "module closure reported the wrong failure: {message}"
    );
}

#[test]
fn malformed_cfg_attribute_fails_closed_in_module_graph_parsing() {
    let fixture = tempfile::tempdir().expect("create malformed cfg fixture");
    let root = fixture.path().join("lib.rs");
    std::fs::write(&root, "#[cfg]\nmod child;\n").expect("write malformed cfg root");
    std::fs::write(fixture.path().join("child.rs"), "fn child() {}\n").expect("write cfg child");

    let result = module_source_closure(fixture.path(), &root, false);
    let Err(message) = result else {
        panic!("module graph accepted malformed cfg attribute");
    };
    assert!(
        message.contains("malformed `cfg` attribute"),
        "module graph reported the wrong cfg failure: {message}"
    );
}

#[test]
fn empty_cfg_predicate_segment_fails_closed() {
    let source = r#"
#[cfg(all(, unix))]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    let result = StructuredSource::parse("fixtures/empty_cfg_segment.rs", source, false).and_then(
        |parsed| {
            parsed.validate_selector(
                0..parsed.tokens.len(),
                DEVICE_SYSTEM_DEFAULT,
                SHARED_LOCK_SELECTOR,
                GuardRequirement::Lexical,
            )
        },
    );
    let Err(message) = result else {
        panic!("source contract accepted an empty cfg predicate segment");
    };
    assert!(
        message.contains("empty cfg predicate"),
        "empty cfg segment reported the wrong failure: {message}"
    );
}

#[test]
fn unretained_command_buffer_api_requires_a_function_lifetime_lock() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let _command_buffer = queue.new_command_buffer_with_unretained_references();
}
"#;
    let parsed = StructuredSource::parse("fixtures/unretained_unguarded.rs", source, true)
        .expect("parse unretained command-buffer fixture");
    let function = parsed
        .test_functions()
        .expect("classify unretained command-buffer test")
        .into_iter()
        .next()
        .expect("find unretained command-buffer test");
    let work = parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER_UNRETAINED, true)
        .expect("classify unretained command-buffer call");
    let error = parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect_err("unguarded unretained command buffer must fail");
    assert!(
        error.contains("no live shared-lock binding encloses this work"),
        "unretained command buffer reported the wrong failure: {error}"
    );
}

#[test]
fn unretained_command_buffer_api_accepts_a_function_lifetime_lock() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let _command_buffer = queue.new_command_buffer_with_unretained_references();
}
"#;
    let parsed = StructuredSource::parse("fixtures/unretained_guarded.rs", source, true)
        .expect("parse guarded unretained command-buffer fixture");
    let function = parsed
        .test_functions()
        .expect("classify guarded unretained command-buffer test")
        .into_iter()
        .next()
        .expect("find guarded unretained command-buffer test");
    let work = parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER_UNRETAINED, true)
        .expect("classify guarded unretained command-buffer call");
    parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect("guarded unretained command buffer satisfies the convention");
}

#[test]
fn cargo_metadata_discovers_nested_and_explicit_target_roots() {
    let fixture = tempfile::tempdir().expect("create Cargo target fixture");
    std::fs::create_dir_all(fixture.path().join("src/bin/nested"))
        .expect("create nested binary directory");
    std::fs::create_dir_all(fixture.path().join("tools"))
        .expect("create explicit target directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        r#"
[package]
name = "target-graph-fixture"
version = "0.0.0"
edition = "2024"

[[example]]
name = "explicit-example"
path = "tools/explicit_example.rs"
"#,
    )
    .expect("write fixture manifest");
    let nested = fixture.path().join("src/bin/nested/main.rs");
    let explicit = fixture.path().join("tools/explicit_example.rs");
    std::fs::write(&nested, "fn main() {}\n").expect("write nested binary");
    std::fs::write(&explicit, "fn main() {}\n").expect("write explicit example");

    let actual = cargo_target_roots(fixture.path(), &["bin", "example"])
        .expect("derive roots from Cargo metadata")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let expected = BTreeSet::from([
        std::fs::canonicalize(nested).expect("canonicalize nested target"),
        std::fs::canonicalize(explicit).expect("canonicalize explicit target"),
    ]);
    assert_eq!(
        actual, expected,
        "Cargo target roots did not match metadata"
    );
}

#[test]
fn cfg_attr_and_tokio_test_functions_are_included_in_raw_dispatch_inventory() {
    let source = r#"
#[cfg_attr(test, test)]
fn cfg_attr_registered_test() {
    let command_buffer = queue.new_command_buffer();
}

#[tokio::test]
async fn tokio_registered_test() {
    let command_buffer = queue.new_command_buffer();
}
"#;
    let parsed = StructuredSource::parse("fixtures/registered_tests.rs", source, true)
        .expect("parse registered test fixture");
    let discovered = parsed
        .test_functions()
        .expect("classify registered test functions")
        .into_iter()
        .filter(|function| {
            !parsed
                .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
                .expect("classify cfg_attr test work")
                .is_empty()
        })
        .map(|function| function.name.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        discovered,
        BTreeSet::from(["cfg_attr_registered_test", "tokio_registered_test"])
    );
}

#[test]
fn const_expression_in_where_clause_does_not_hide_unguarded_dispatch() {
    let source = r#"
struct Queue;
impl Queue { fn new_command_buffer(&self) {} }
trait Marker<const N: usize> {}
impl Marker<1> for () {}

#[test]
fn raw_dispatch() where (): Marker<{ 1 }> {
    let queue = Queue;
    let _command_buffer = queue.new_command_buffer();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/where_const_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn const_expression_in_return_type_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() -> Result<(), Marker<{ 1 }>> {
    let _command_buffer = queue.new_command_buffer();
    Ok(())
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/return_const_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn labeled_loop_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() {
    'gpu: loop {
        let _command_buffer = queue.new_command_buffer();
        break 'gpu;
    }
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/labeled_loop_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn closure_does_not_hide_unguarded_dispatch() {
    let source = r#"
#[test]
fn raw_dispatch() {
    let drive = || queue.new_command_buffer();
    drive();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/closure_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn unknown_test_registration_attribute_fails_closed() {
    let source = r#"
#[custom_runtime::test]
async fn raw_dispatch() {
    let queue = Queue;
    let _command_buffer = queue.new_command_buffer();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "tests/custom_runtime_dispatch.rs",
        "unclassifiable test function attribute `custom_runtime::test`",
    );
}

#[test]
fn included_non_rust_source_in_test_scope_fails_closed() {
    let fixture = tempfile::tempdir().expect("create include fixture");
    let source_path = fixture.path().join("dispatch_test.rs");
    let included_path = fixture.path().join("included_dispatch.inc");
    let source = r#"
#[test]
fn raw_dispatch() {
    include!("included_dispatch.inc");
}
"#;
    std::fs::write(&source_path, source).expect("write include fixture source");
    std::fs::write(
        &included_path,
        "let _command_buffer = queue.new_command_buffer();\n",
    )
    .expect("write included non-Rust source");

    assert_command_buffer_fixture_rejected_with(
        &std::fs::read_to_string(&source_path).expect("read include fixture source"),
        "tests/dispatch_test.rs",
        "unclassifiable include! in test-bearing scope",
    );
}

#[test]
fn compiler_selected_non_rust_module_is_inside_the_test_boundary() {
    let fixture = tempfile::tempdir().expect("create non-Rust module fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[path = \"dispatch.inc\"]\nmod dispatch;\n",
    )
    .expect("write fixture crate root");
    std::fs::write(
        src.join("dispatch.inc"),
        r#"
#[test]
fn raw_dispatch() {
    let _command_buffer = queue.new_command_buffer();
}
"#,
    )
    .expect("write compiler-selected non-Rust module");

    let sources = parsed_module_closure(fixture.path(), &src.join("lib.rs"), true)
        .expect("parse compiler-selected module closure");
    let child = sources
        .iter()
        .find(|source| source.relative.ends_with("src/dispatch.inc"))
        .expect("non-Rust module belongs to the compiler-selected closure");
    let function = child
        .parsed
        .test_functions()
        .expect("classify non-Rust module tests")
        .into_iter()
        .find(|function| function.name == "raw_dispatch")
        .expect("find raw dispatch test");
    let work = child
        .parsed
        .call_sites(function.body.clone(), NEW_COMMAND_BUFFER, true)
        .expect("classify non-Rust module work");
    let error = child
        .parsed
        .validate_work_sites(
            &work,
            LOCAL_LOCK_SELECTOR,
            GuardRequirement::Function {
                closing_brace: function.body.end,
            },
        )
        .expect_err("unguarded non-Rust module work must fail");
    assert!(
        error.contains("no live shared-lock binding encloses this work"),
        "non-Rust module reported the wrong failure: {error}"
    );
}

#[test]
fn cfg_attr_generated_module_path_fails_closed() {
    let fixture = tempfile::tempdir().expect("create cfg_attr path fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg_attr(all(), path = \"dispatch.inc\")]\nmod dispatch;\n",
    )
    .expect("write fixture crate root");
    std::fs::write(src.join("dispatch.inc"), "").expect("write cfg_attr-selected module source");

    let error = module_source_closure(fixture.path(), &src.join("lib.rs"), true)
        .err()
        .expect("cfg_attr-generated path must fail closed");
    assert!(
        error.contains("unclassifiable cfg_attr-generated module path"),
        "cfg_attr module path reported the wrong failure: {error}"
    );
}

#[test]
fn forwarded_receiver_and_identifier_macro_fails_closed() {
    let source = r#"
macro_rules! raw_dispatch {
    ($receiver:expr, $method:ident) => {
        $receiver.$method()
    };
}

#[test]
fn forwarded_dispatch() {
    let _command_buffer = raw_dispatch!(queue, new_command_buffer);
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/forwarded_macro_dispatch.rs",
        "unclassifiable macro invocation `raw_dispatch!` in test-bearing scope",
    );
}

#[test]
fn guard_lifetime_rejects_an_immediate_drop_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    drop(gpu_guard);
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped before Metal work");
}

#[test]
fn guard_lifetime_rejects_a_wildcard_that_drops_the_guard_immediately() {
    let source = r#"
fn measurement() {
    let _ = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a wildcard guard binding");
}

#[test]
fn guard_lifetime_rejects_a_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_stacked_cfg_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a stacked cfg-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_an_active_cfg_attr_elided_acquisition() {
    let source = r#"
fn measurement() {
    #[cfg_attr(all(), cfg(any()))]
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "an active cfg_attr-elided guard acquisition");
}

#[test]
fn guard_lifetime_rejects_a_move_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let moved_guard = gpu_guard;
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a moved guard binding");
}

#[test]
fn guard_lifetime_rejects_shadowing_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let gpu_guard = ();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a shadowed guard binding");
}

#[test]
fn guard_lifetime_rejects_a_guard_in_a_nested_scope() {
    let source = r#"
fn measurement() {
    {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a guard dropped with a nested scope");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_function() {
    let source = r#"
#[cfg(any())]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_cfg.rs");
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_module() {
    let source = r#"
#[cfg(any())]
mod hidden {
    fn measurement() {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_module.rs");
}

#[test]
fn module_closure_excludes_an_out_of_line_cfg_elided_module() {
    let fixture = tempfile::tempdir().expect("create module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create fixture source directory");
    std::fs::write(src.join("lib.rs"), "#[cfg(any())]\npub mod child;\n")
        .expect("write fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write fixture module child");
    let sources = module_source_closure(fixture.path(), &src.join("lib.rs"), false)
        .expect("classify cfg-elided module closure");
    assert!(
        sources.iter().all(|source| source.path != child),
        "cfg-elided child entered the compiler-selected closure"
    );
}

#[test]
fn module_closure_excludes_an_out_of_line_module_in_a_cfg_elided_parent() {
    let fixture = tempfile::tempdir().expect("create nested module-context fixture");
    let src = fixture.path().join("src");
    std::fs::create_dir(&src).expect("create nested fixture source directory");
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg(any())]\nmod hidden {\n    #[path = \"child.rs\"]\n    pub mod child;\n}\n",
    )
    .expect("write nested fixture module parent");
    let child = src.join("child.rs");
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    std::fs::write(&child, source).expect("write nested fixture module child");
    let sources = module_source_closure(fixture.path(), &src.join("lib.rs"), false)
        .expect("classify nested cfg-elided module closure");
    assert!(
        sources.iter().all(|source| source.path != child),
        "child below a cfg-elided inline parent entered the compiler-selected closure"
    );
}

#[test]
fn guard_lifetime_rejects_an_enclosing_cfg_elided_block() {
    let source = r#"
fn measurement() {
    #[cfg(any())]
    unsafe {
        let gpu_guard = lattice_inference::measurement::gpu_test_lock();
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/enclosing_false_block.rs");
}

#[test]
fn guard_lifetime_rejects_a_recursively_cfg_elided_function() {
    let source = r#"
#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/nested_false_cfg_attr.rs");
}

#[test]
fn guard_lifetime_checks_every_work_occurrence_after_a_false_gated_decoy() {
    let source = r#"
#[cfg(any())]
fn guarded_decoy() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}

fn live_work() {
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/false_gated_decoy.rs");
}

#[test]
fn guard_lifetime_ignores_a_comment_decoy_before_live_work() {
    let source = r#"
fn measurement() {
    {
        // Device::system_default()
    }
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_accepted(source, "fixtures/comment_decoy.rs");
}

#[test]
fn raw_dispatch_rejects_source_trivia_before_call_parentheses() {
    let source = r#"
#[test]
fn trivia_dispatch() {
    let command_buffer = queue.new_command_buffer /* comment */ ();
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/trivia_dispatch.rs",
        "no live shared-lock binding encloses this work",
    );
}

#[test]
fn guard_lifetime_uses_braces_for_an_irregularly_indented_nested_scope() {
    let source = r#"
fn measurement() {
    {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
   }
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "fixtures/irregular_nested_guard.rs");
}

#[test]
fn raw_dispatch_rejects_unclassifiable_macro_work() {
    let source = r#"
#[test]
fn macro_dispatch() {
    let gpu_guard = gpu_test_lock();
    dispatch!(queue.new_command_buffer());
}
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/macro_dispatch.rs",
        "unclassifiable macro invocation `dispatch!` in test-bearing scope",
    );
}

#[test]
fn raw_dispatch_rejects_macro_generated_test_work() {
    let source = r#"
macro_rules! raw_test {
    () => {
        #[test]
        fn generated() {
            let command_buffer = queue.new_command_buffer();
        }
    };
}
raw_test!();
"#;
    assert_command_buffer_fixture_rejected_with(
        source,
        "src/forward/macro_generated_test.rs",
        "unclassifiable macro invocation `raw_test!` in test-bearing scope",
    );
}

#[test]
fn guard_lifetime_rejects_stacked_cfg_separated_by_docs_and_blank_lines() {
    let source = r#"
fn measurement() {
    #[cfg(any())]

    /// The active Metal acquisition.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "stacked cfg with intervening documentation");
}

#[test]
fn guard_lifetime_rejects_a_raw_identifier_binding() {
    let source = r#"
fn measurement() {
    let r#gpu_guard = lattice_inference::measurement::gpu_test_lock();
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a raw-identifier guard binding");
}

#[test]
fn guard_lifetime_rejects_a_let_else_derived_binding() {
    let source = r#"
fn measurement() {
    let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) else {
        return;
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a let-else-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_an_if_let_derived_binding() {
    let source = r#"
fn measurement() {
    if let Some(gpu_guard) = Some(lattice_inference::measurement::gpu_test_lock()) {
        let _device = Device::system_default();
    }
}
"#;
    assert_guard_fixture_rejected(source, "an if-let-derived guard binding");
}

#[test]
fn guard_lifetime_rejects_a_match_derived_binding() {
    let source = r#"
fn measurement() {
    let gpu_guard = match () {
        () => lattice_inference::measurement::gpu_test_lock(),
    };
    let _device = Device::system_default();
}
"#;
    assert_guard_fixture_rejected(source, "a match-derived guard binding");
}

/// Accumulates the construction-site inventory for
/// `metal_qwen35_state_construction_tests_use_function_lifetime_lock_bindings`,
/// keeping the exemption bookkeeping (discovered/classified/used/moved) in
/// one place instead of duplicating it across the lib and non-lib scan
/// branches.
struct ConstructionTally<'a> {
    exemptions: &'a BTreeMap<String, &'a ConstructionExemption>,
    discovered: BTreeSet<String>,
    classified: BTreeSet<String>,
    used_exemptions: BTreeSet<String>,
    moved_exemptions: Vec<String>,
}

impl<'a> ConstructionTally<'a> {
    /// Records one discovered construction site under its stable key. When
    /// the site matches a recorded exemption, classifies it, marks the
    /// exemption used, checks the position tripwire, and returns `None` (the
    /// caller must skip the live-lock check). Otherwise returns `Some(site)`
    /// for the caller to run the live-lock check against.
    fn record(
        &mut self,
        parsed: &StructuredSource,
        site_prefix: &str,
        construction: usize,
        selector: CallSelector,
        ordinals: &mut BTreeMap<(String, String), usize>,
    ) -> Option<String> {
        let function_path = parsed.enclosing_function_path(construction);
        let counter = ordinals
            .entry((function_path, selector.label()))
            .or_insert(0);
        *counter += 1;
        let site = format!(
            "{site_prefix}=>{}",
            parsed.stable_construction_key(construction, selector, *counter)
        );
        self.discovered.insert(site.clone());
        let Some(exemption) = self.exemptions.get(&site) else {
            return Some(site);
        };
        self.classified.insert(site.clone());
        self.used_exemptions.insert(site.clone());
        let current_position = parsed.source_site_label(construction);
        if current_position != exemption.recorded_position {
            self.moved_exemptions.push(format!(
                "{site}: recorded position `{}` no longer matches current position \
                 `{current_position}`; confirm this exemption still describes the same work \
                 (this is not a locking violation)",
                exemption.recorded_position
            ));
        }
        None
    }
}

#[test]
fn metal_qwen35_state_construction_tests_use_function_lifetime_lock_bindings() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exemptions = CONSTRUCTION_EXEMPTIONS
        .iter()
        .map(|exemption| {
            assert!(
                !exemption.reason.trim().is_empty(),
                "construction exemption {} needs a justification",
                exemption.site
            );
            (exemption.site.to_string(), exemption)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        exemptions.len(),
        CONSTRUCTION_EXEMPTIONS.len(),
        "construction exemptions must not contain duplicate sites"
    );

    let mut tally = ConstructionTally {
        exemptions: &exemptions,
        discovered: BTreeSet::new(),
        classified: BTreeSet::new(),
        used_exemptions: BTreeSet::new(),
        moved_exemptions: Vec::new(),
    };
    let mut violations = Vec::new();
    for target in cargo_targets(manifest_dir, &["bench", "example", "bin", "test", "lib"])
        .unwrap_or_else(|reason| panic!("{reason}"))
    {
        let target_relative = target
            .path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        let sources = parsed_module_closure(
            manifest_dir,
            &target.path,
            matches!(target.kind.as_str(), "test" | "lib"),
        )
        .unwrap_or_else(|reason| panic!("{reason}"));
        let caller_owned = BENCH_HARNESSES
            .iter()
            .find(|contract| contract.path == target_relative && !contract.calls.is_empty());
        if let Some(contract) = caller_owned {
            validate_bench_harness_sources(&sources, contract)
                .unwrap_or_else(|reason| panic!("{}: {reason}", contract.path));
        }
        for source in &sources {
            let site_prefix = format!("{}:{}:{}", target.kind, target.name, target_relative);
            let mut ordinals: BTreeMap<(String, String), usize> = BTreeMap::new();

            if target.kind == "lib" {
                for function in source
                    .parsed
                    .test_functions()
                    .unwrap_or_else(|reason| panic!("{reason}"))
                {
                    for (construction, selector) in source
                        .parsed
                        .construction_sites(function.body.clone())
                        .unwrap_or_else(|reason| panic!("{reason}"))
                    {
                        let Some(site) = tally.record(
                            &source.parsed,
                            &site_prefix,
                            construction,
                            selector,
                            &mut ordinals,
                        ) else {
                            continue;
                        };
                        match source.parsed.validate_work_sites(
                            &[construction],
                            LOCAL_LOCK_SELECTOR,
                            GuardRequirement::Function {
                                closing_brace: function.body.end,
                            },
                        ) {
                            Ok(()) => {
                                tally.classified.insert(site);
                            }
                            Err(reason) => violations.push(format!("{site}: {reason}")),
                        }
                    }
                }
                continue;
            }

            for (construction, selector) in source
                .parsed
                .construction_sites(0..source.parsed.tokens.len())
                .unwrap_or_else(|reason| panic!("{reason}"))
            {
                let Some(site) = tally.record(
                    &source.parsed,
                    &site_prefix,
                    construction,
                    selector,
                    &mut ordinals,
                ) else {
                    continue;
                };
                if caller_owned.is_some() {
                    // The same gate verifies the exact construction inventory,
                    // all caller edges, live owner guards, and registration.
                    tally.classified.insert(site);
                    continue;
                }
                match source.parsed.validate_work_sites(
                    &[construction],
                    SHARED_LOCK_SELECTOR,
                    GuardRequirement::Lexical,
                ) {
                    Ok(()) => {
                        tally.classified.insert(site);
                    }
                    Err(reason) => violations.push(format!("{site}: {reason}")),
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "MetalQwen35State construction sites without a live shared-lock binding:\n{}",
        violations.join("\n")
    );
    assert!(
        tally.moved_exemptions.is_empty(),
        "construction exemptions recorded a position that no longer matches (not a locking \
         violation — update recorded_position once the move is confirmed intentional):\n{}",
        tally.moved_exemptions.join("\n")
    );

    let expected_exemptions = exemptions.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        tally.used_exemptions, expected_exemptions,
        "construction exemptions must identify current exact sites"
    );
    assert_construction_inventory_classified(&tally.discovered, &tally.classified);
}

#[test]
fn construction_inventory_comparison_rejects_an_unclassified_site() {
    let discovered = BTreeSet::from(["src/bin/known.rs".to_string(), "src/bin/new.rs".to_string()]);
    let classified = BTreeSet::from(["src/bin/known.rs".to_string()]);

    let result = std::panic::catch_unwind(|| {
        assert_construction_inventory_classified(&discovered, &classified);
    });
    assert!(
        result.is_err(),
        "inventory comparison accepted an unclassified site"
    );
}

#[test]
fn rust_targets_have_one_gpu_lock_definition_and_path() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let definition = ["fn gpu", "_test_lock"].concat();
    let lock_path = ["/tmp/lion-metal", "-gpu-test.lock"].concat();
    let mut definitions = BTreeSet::new();
    let mut paths = BTreeSet::new();

    for path in rust_sources_under(manifest_dir) {
        let source = std::fs::read_to_string(&path).expect("read Rust source");
        let relative = path
            .strip_prefix(manifest_dir)
            .expect("source under manifest directory")
            .to_string_lossy()
            .into_owned();
        if source.contains(&definition) {
            definitions.insert(relative.clone());
        }
        if source.contains(&lock_path) {
            paths.insert(relative);
        }
    }

    let only_shared_module = BTreeSet::from(["src/measurement.rs".to_string()]);
    assert_eq!(
        definitions, only_shared_module,
        "GPU lock behavior must have exactly one Rust definition"
    );
    assert_eq!(
        paths, only_shared_module,
        "the fleet GPU lock path must have exactly one Rust source of truth"
    );
}

#[test]
fn prior_lock_owners_and_existing_bench_use_the_shared_module() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let callers = [
        (
            "src/forward/metal.rs",
            "use crate::measurement::gpu_test_lock;",
        ),
        (
            "src/forward/metal_qwen35.rs",
            "use crate::measurement::gpu_test_lock;",
        ),
        (
            "tests/vision_s3b_vit_metal_gate_test.rs",
            "use lattice_inference::measurement::gpu_test_lock;",
        ),
        ("src/bin/bench_gdn_prefill_ab.rs", SHARED_LOCK_CALL),
    ];

    for (relative, shared_reference) in callers {
        let source =
            std::fs::read_to_string(manifest_dir.join(relative)).expect("read migrated caller");
        assert!(
            source.contains(shared_reference),
            "{relative} no longer references the one shared GPU lock"
        );
    }
}
