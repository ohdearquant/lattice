use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::{Attribute, ItemFn, Meta, Token};

#[path = "support/source_graph.rs"]
mod source_graph;
use source_graph::*;

#[path = "support/measurement_lock_analyzer.rs"]
mod measurement_lock_analyzer;
#[path = "support/measurement_lock_bench_harness.rs"]
mod measurement_lock_bench_harness;
#[path = "support/measurement_lock_construction.rs"]
mod measurement_lock_construction;
#[path = "support/measurement_lock_generate.rs"]
mod measurement_lock_generate;
#[path = "support/measurement_lock_helpers.rs"]
mod measurement_lock_helpers;
#[path = "support/measurement_lock_packages.rs"]
mod measurement_lock_packages;
#[path = "support/measurement_lock_receipt_reader.rs"]
mod measurement_lock_receipt_reader;
use measurement_lock_receipt_reader::{RECEIPT_READER_REASON, lock_source_inventory};

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
/// The package that defines the lock; `lattice-inference` re-exports it as
/// `measurement::gpu_test_lock`.
const LOCK_DEFINITION_PACKAGE: &str = "../metal-runtime";
const RAW_HARNESS_ENTRYPOINTS: &[(&str, &str, CallSelector)] = &[
    (
        ".",
        "benches/decode_attn_bench.rs",
        CallSelector::Path(&["bench_flash_decode"]),
    ),
    (
        ".",
        "benches/topk_readback.rs",
        CallSelector::Path(&["bench_topk_parity"]),
    ),
    (
        ".",
        "examples/bench_concurrent.rs",
        CallSelector::Path(&["run"]),
    ),
    (".", "examples/bench_dispatch.rs", DEVICE_SYSTEM_DEFAULT),
    (".", "examples/bench_dispatch2.rs", DEVICE_SYSTEM_DEFAULT),
    (
        ".",
        "examples/bench_gemm_shapes.rs",
        CallSelector::Path(&["run"]),
    ),
    (".", "examples/bench_mps_gemm.rs", DEVICE_SYSTEM_DEFAULT),
    (".", "examples/bench_simdgroup.rs", DEVICE_SYSTEM_DEFAULT),
    (
        ".",
        "examples/profile_metal_decode.rs",
        DEVICE_SYSTEM_DEFAULT,
    ),
];
const ADDITIONAL_GUARDED_ENTRYPOINTS: &[(&str, &str, CallSelector)] = &[
    (".", "tests/metal_runtime_buffers.rs", DEVICE_SYSTEM_DEFAULT),
    (
        ".",
        "examples/bench_embed_quality.rs",
        CallSelector::Path(&["run_bench"]),
    ),
    (
        ".",
        "examples/bench_metal.rs",
        CallSelector::Path(&["MetalForwardPass", "new"]),
    ),
];
/// Methods that drive Metal GPU work directly on an already-constructed
/// `MetalQwen35State`, with no raw selector (`RAW_GPU_SELECTORS`) and no
/// constructor call (`CONSTRUCTION_SELECTORS`) anywhere at the call site
/// itself. A target reaching the GPU only through one of these is invisible
/// to both of those lexical scans (#1527); every name below was confirmed,
/// file by file, to be used only on a Metal state receiver and only inside
/// the same function as its guard, so a bare method-name match cannot pick
/// up an unrelated type's same-named method by accident.
const STATE_WORK_SELECTORS: &[CallSelector] = &[
    CallSelector::Method("forward_step"),
    CallSelector::Method("forward_prefill"),
    CallSelector::Method("forward_step_with_hidden"),
    CallSelector::Method("forward_prefill_with_hidden"),
    CallSelector::Method("forward_prefill_all_logits"),
];
/// `examples/` and `src/bin/` targets whose module closure calls a
/// `STATE_WORK_SELECTORS` method directly, restricted to targets not already
/// owned by `RAW_HARNESS_ENTRYPOINTS` or `ADDITIONAL_GUARDED_ENTRYPOINTS`.
/// Closes the same admission gap #1524 closed for `benches/`
/// (`validate_bench_harness_inventory`) on the `examples/`+`src/bin/` side:
/// `state_driven_metal_measurement_targets_use_live_lock_bindings` asserts
/// this list against the lexical scan, so a target added or removed from
/// either side without updating the other fails the test — "found, but never
/// examined" cannot happen silently. This list is deliberately narrower than
/// the full `TARGETS_WITH_RECOGNIZED_METAL_MARKERS` classification: several
/// targets in that broader set drive the GPU only through a higher-level
/// wrapper (`generate`, `chat_completion`, `compute_perplexity`, the
/// `QwenModel` `encode*` family) or through a library helper reached across
/// a module boundary, and are out of scope for this lexical, single-file
/// call-site check (named as a residual gap, not silently swept in).
const TARGETS_WITH_STATE_DRIVEN_WORK: &[(&str, &str)] = &[
    (".", "examples/bench_gdn_decode.rs"),
    (".", "examples/bench_gdn_prefill_ab.rs"),
    (".", "examples/bench_gdn_state.rs"),
    (".", "examples/bench_metal_forward_allocation.rs"),
    (".", "examples/bench_pruning.rs"),
    (".", "examples/bench_q4_prefill.rs"),
    (".", "examples/bench_q8_prefill.rs"),
    (".", "examples/bench_quality.rs"),
    (".", "examples/decode_profile.rs"),
    (".", "examples/profile_metal.rs"),
    (".", "examples/rss_autorelease_probe.rs"),
    (".", "src/bin/bench_logit_dump.rs"),
];
/// A same-file helper-mediated exception within `TARGETS_WITH_STATE_DRIVEN_WORK`,
/// the `STATE_WORK_SELECTORS` analogue of `ADDITIONAL_GUARDED_ENTRYPOINTS`
/// (whose own inventory test already names "helper-mediated" as a reviewed
/// category, `cargo_target_lexical_metal_marker_inventory_is_explicit`).
/// `examples/bench_gdn_prefill_ab.rs::run_interleaved` holds the shared lock
/// for its whole body and calls `run_one` (a private helper, single caller,
/// confirmed by grep) synchronously several times before releasing it; the
/// low-level `forward_prefill` call sits in `run_one`'s own body, one lexical
/// scope removed from the guard, so a direct `STATE_WORK_SELECTORS` scan
/// cannot see it as enclosed even though every real invocation is. The same
/// file's unrelated `run` function calls `forward_prefill` directly under its
/// own guard and keeps going through the ordinary path below; only the
/// `run_one`-mediated call sites are rerouted to validate the entrypoint
/// (`run_one(...)`) instead of the low-level method inside it.
const STATE_WORK_HELPER_ENTRYPOINTS: &[(&str, &str, CallSelector)] = &[(
    ".",
    "examples/bench_gdn_prefill_ab.rs",
    CallSelector::Path(&["run_one"]),
)];

struct GenerateWorkTarget {
    package: &'static str,
    path: &'static str,
    owner: &'static str,
    receiver: &'static str,
    method: &'static str,
    calls: usize,
}

impl GenerateWorkTarget {
    fn key(&self) -> String {
        package_path_key(self.package, self.path)
    }
}

// Only these reviewed synchronous receivers and methods are covered; other
// generation methods, mixed CPU/Metal targets, and cross-module wrapper calls
// need separate contracts.
const GENERATE_WORK_TARGETS: &[GenerateWorkTarget] = &[
    GenerateWorkTarget {
        package: ".",
        path: "examples/bench_persistent_state.rs",
        owner: "run",
        receiver: "state",
        method: "generate",
        calls: 2,
    },
    GenerateWorkTarget {
        package: ".",
        path: "examples/bench_decode_streaming.rs",
        owner: "run",
        receiver: "metal",
        method: "generate_streaming_with_cancel",
        calls: 1,
    },
    GenerateWorkTarget {
        package: ".",
        path: "src/bin/bench_decode_slopefit.rs",
        owner: "run",
        receiver: "metal",
        method: "generate",
        calls: 2,
    },
    GenerateWorkTarget {
        package: ".",
        path: "src/bin/dump_quarot_q4_golden.rs",
        owner: "run",
        receiver: "metal",
        method: "generate",
        calls: 1,
    },
];

struct BenchHarnessCall {
    selector: CallSelector,
    callers: &'static [&'static str],
    guarded: bool,
}

struct BenchHarness {
    package: &'static str,
    path: &'static str,
    registered: &'static [&'static str],
    calls: &'static [BenchHarnessCall],
    constructions: &'static [(&'static str, CallSelector)],
}

impl BenchHarness {
    fn key(&self) -> String {
        package_path_key(self.package, self.path)
    }
}

// These are finite, reviewed caller chains, not Rust name resolution. The
// syntax check below rejects aliases, deferred calls, and other spellings of
// these names rather than silently losing the ownership proof.
const BENCH_HARNESSES: &[BenchHarness] = &[
    BenchHarness {
        package: ".",
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
        package: ".",
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
        package: ".",
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
        package: ".",
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
        package: ".",
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
        package: ".",
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
const TARGETS_WITH_RECOGNIZED_METAL_MARKERS: &[(&str, &str)] = &[
    (".", "benches/cross_turn_prefix_cache_bench.rs"),
    (".", "benches/decode_attn_bench.rs"),
    (".", "benches/lm_head_bench.rs"),
    (".", "benches/metal_decode_bench.rs"),
    (".", "benches/mtp_decode.rs"),
    (".", "benches/topk_readback.rs"),
    (".", "examples/bench_concurrent.rs"),
    (".", "examples/bench_decode_streaming.rs"),
    (".", "examples/bench_dispatch.rs"),
    (".", "examples/bench_dispatch2.rs"),
    (".", "examples/bench_embed_quality.rs"),
    (".", "examples/bench_embedding.rs"),
    (".", "examples/bench_gdn_decode.rs"),
    (".", "examples/bench_gdn_prefill_ab.rs"),
    (".", "examples/bench_gdn_state.rs"),
    (".", "examples/bench_gemm_shapes.rs"),
    (".", "examples/bench_gpu.rs"),
    (".", "examples/bench_metal.rs"),
    (".", "examples/bench_metal_forward_allocation.rs"),
    (".", "examples/bench_mps_gemm.rs"),
    (".", "examples/bench_persistent_state.rs"),
    (".", "examples/bench_profile.rs"),
    (".", "examples/bench_pruning.rs"),
    (".", "examples/bench_q4_prefill.rs"),
    (".", "examples/bench_q8_prefill.rs"),
    (".", "examples/bench_quality.rs"),
    ("../serve", "examples/bench_serve_prepare.rs"),
    (".", "examples/bench_simdgroup.rs"),
    (".", "examples/bench_stability.rs"),
    (".", "examples/bench_suite.rs"),
    (".", "examples/decode_profile.rs"),
    (".", "examples/dump_embeddinggemma2.rs"),
    (".", "examples/dump_embeddinggemma2_vision.rs"),
    (".", "examples/layer_sweep.rs"),
    (".", "examples/profile_metal.rs"),
    (".", "examples/profile_metal_decode.rs"),
    (".", "examples/rss_autorelease_probe.rs"),
    (".", "src/bin/backfill_qwen3.rs"),
    (".", "src/bin/bench_decode_ab.rs"),
    (".", "src/bin/bench_decode_slopefit.rs"),
    (".", "src/bin/bench_gdn_prefill_ab.rs"),
    (".", "src/bin/bench_logit_dump.rs"),
    (".", "src/bin/bench_lora_mixture.rs"),
    (".", "src/bin/bench_vision_prefill_ab.rs"),
    (".", "src/bin/chat_metal.rs"),
    (".", "src/bin/dump_quarot_q4_golden.rs"),
    (".", "src/bin/eval_perplexity.rs"),
    (".", "src/bin/gramperf_profile.rs"),
    ("../serve", "src/bin/lattice/main.rs"),
    (".", "src/bin/ppl_metal.rs"),
];
const TARGETS_WITHOUT_RECOGNIZED_METAL_MARKERS: &[(&str, &str)] = &[
    (".", "benches/attention_dispatch_bench.rs"),
    (".", "benches/attn_opt_bench.rs"),
    (".", "benches/backward_stage0.rs"),
    (".", "benches/differential_attention_bench.rs"),
    (".", "benches/e2e_bench.rs"),
    (".", "benches/elementwise_bench.rs"),
    (".", "benches/elementwise_cpu_bench.rs"),
    (".", "benches/f16_convert_bench.rs"),
    (".", "benches/gated_attention_bench.rs"),
    (".", "benches/grammar_mask_bench.rs"),
    (".", "benches/inference_bench.rs"),
    (".", "benches/inference_perf.rs"),
    (".", "benches/kv_cache_f16_bench.rs"),
    (".", "benches/kv_cache_layout_bench.rs"),
    (".", "benches/metrics_bench.rs"),
    (".", "benches/native_sparse_attention_bench.rs"),
    (".", "benches/pruning_bench.rs"),
    (".", "benches/quarot_hadamard_bench.rs"),
    (".", "benches/tokenizer_bench.rs"),
    (".", "examples/bench_gdn.rs"),
    ("../serve", "examples/bench_serve_http.rs"),
    (".", "examples/diff_attn_layer23.rs"),
    (".", "examples/diff_gdn_layer.rs"),
    (".", "examples/ernie45_trace_dump.rs"),
    (".", "examples/load_cross_encoder.rs"),
    (".", "src/bin/bench_gemma4_driver_pair.rs"),
    (".", "src/bin/gramtime_profile.rs"),
    ("../serve", "src/bin/lattice_serve/main.rs"),
    ("../serve", "src/bin/moe_admission_sim.rs"),
    ("../serve", "src/bin/quantize_q4.rs"),
    ("../serve", "src/bin/quantize_quarot.rs"),
    (".", "src/bin/qwen35_debug.rs"),
    (".", "src/bin/qwen35_generate.rs"),
];
const IN_CRATE_COMMAND_BUFFER_TESTS: &[(&str, &str)] = &[
    (
        "../models/qwen3-embedding",
        "src/metal.rs::rms_norm_matches_pre_854_oracle_on_finite_input",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests.rs::test_gemv_decode_numerical",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/decode_attention.rs::test_gpu_argmax_parity_k1",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/gdn_chunked_scan.rs::decode_attention_reference_fails_closed_on_nan_q",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q3.rs::gemv_q3_decode_matches_cpu_reference_across_shapes",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q3.rs::gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q3.rs::gemm_q3_tiled_differential_fails_closed_on_nan_scale",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q3.rs::gemv_q3_decode_mutation_sensitive_high_plane_bit",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/checkpoint_refusals.rs::lora_gemv_kernel_matches_cpu_reference",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/checkpoint_refusals.rs::load_adapter_and_dispatch_lora_if_active",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/lora.rs::lora_rows_dispatch_matches_cpu_reference_across_rows_ranks_and_strides",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/checkpoint_refusals.rs::forced_non_apple7_q4_gemm_fallback_dispatches_and_matches_reference",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/dispatch.rs::dispatch_matmul_q4_writes_all_rows",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q4_signed_scale.rs::gemv_q4_decode_honours_negative_scales",
    ),
    (
        "../models/qwen35",
        "src/forward/metal_qwen35/inner/tests/q4_signed_scale.rs::lm_head_block_topk_q4_honours_negative_scales",
    ),
];
const CONSTRUCTION_SELECTORS: &[CallSelector] = &[
    CallSelector::Path(&["MetalEmbeddingGemma2State", "new"]),
    CallSelector::Path(&["MetalErnie45State", "new"]),
    CallSelector::Path(&["MetalQwen35State", "new"]),
    CallSelector::Path(&["MetalQwen35State", "from_q4_dir"]),
    // QwenModel::from_directory attempts MetalForwardPass::new internally
    // (crates/models/qwen3-embedding/src/lib.rs) whenever Metal is available, so
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
    package: &'static str,
    site: &'static str,
    recorded_position: &'static str,
    reason: &'static str,
}

const CONSTRUCTION_EXEMPTIONS: &[ConstructionExemption] = &[
    ConstructionExemption {
        package: ".",
        site: "example:bench_concurrent:examples/bench_concurrent.rs=>examples/bench_concurrent.rs::run::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "examples/bench_concurrent.rs:444:27",
        reason: "run state1 construction is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        package: ".",
        site: "example:bench_concurrent:examples/bench_concurrent.rs=>examples/bench_concurrent.rs::run::MetalQwen35State::from_q4_dir()#2",
        recorded_position: "examples/bench_concurrent.rs:474:27",
        reason: "run state2 construction is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        package: ".",
        site: "example:bench_embed_quality:examples/bench_embed_quality.rs=>examples/bench_embed_quality.rs::run_bench::QwenModel::from_directory()#1",
        recorded_position: "examples/bench_embed_quality.rs:136:52",
        reason: "run_bench QwenModel::from_directory is reached only below main's checked live guard; arbitrary call-graph proof is outside this lexical contract",
    },
    ConstructionExemption {
        package: "../serve",
        site: "example:bench_serve_prepare:examples/bench_serve_prepare.rs=>examples/bench_serve_prepare.rs::run_metal::MetalQwen35State::new()#1",
        recorded_position: "examples/bench_serve_prepare.rs:688:47",
        reason: "run_metal constructs the state inside the worker loader closure; the shared lock is acquired on the worker thread before construction by bench_support::spawn_with_measurement_lock, so no caller-side lexical binding encloses this site",
    },
    ConstructionExemption {
        package: "../serve",
        site: "example:bench_serve_prepare:examples/bench_serve_prepare.rs=>examples/bench_serve_prepare.rs::run_metal::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "examples/bench_serve_prepare.rs:698:47",
        reason: "run_metal constructs the state inside the worker loader closure; the shared lock is acquired on the worker thread before construction by bench_support::spawn_with_measurement_lock, so no caller-side lexical binding encloses this site",
    },
    ConstructionExemption {
        package: ".",
        site: "bin:backfill_qwen3:src/bin/backfill_qwen3.rs=>src/bin/backfill_qwen3.rs::main::QwenModel::from_directory()#1",
        recorded_position: "src/bin/backfill_qwen3.rs:37:28",
        reason: "one-time DB embedding migration over an unbounded row count belongs to a long-running batch process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        package: ".",
        site: "bin:chat_metal:src/bin/chat_metal.rs=>src/bin/chat_metal.rs::run::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/chat_metal.rs:474:39",
        reason: "run Q4 initialization belongs to a long-running interactive process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        package: ".",
        site: "bin:chat_metal:src/bin/chat_metal.rs=>src/bin/chat_metal.rs::run::MetalQwen35State::new()#1",
        recorded_position: "src/bin/chat_metal.rs:499:39",
        reason: "run safetensors initialization belongs to a long-running interactive process outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        package: "../serve",
        site: "bin:lattice:src/bin/lattice/main.rs=>src/bin/lattice/chat.rs::MetalChatBackend::load::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/bin/lattice/chat.rs:53:81",
        reason: "MetalChatBackend::load belongs to an interactive or one-shot product process outside the measurement-harness contract",
    },
    ConstructionExemption {
        package: "../serve",
        site: "bin:lattice:src/bin/lattice/main.rs=>src/serving_provider/providers/qwen/lattice_worker.rs::open::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/serving_provider/providers/qwen/lattice_worker.rs:64:47",
        reason: "the Qwen lattice worker initializes a long-running server worker outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        package: "../serve",
        site: "bin:lattice_serve:src/bin/lattice_serve/main.rs=>src/serving_provider/providers/qwen/worker.rs::load_model::MetalQwen35State::from_q4_dir()#1",
        recorded_position: "src/serving_provider/providers/qwen/worker.rs:116:43",
        reason: "load_model Q4 initialization belongs to a long-running server outside the bounded measurement-harness contract",
    },
    ConstructionExemption {
        package: "../serve",
        site: "bin:lattice_serve:src/bin/lattice_serve/main.rs=>src/serving_provider/providers/qwen/worker.rs::load_model::MetalQwen35State::new()#1",
        recorded_position: "src/serving_provider/providers/qwen/worker.rs:135:43",
        reason: "load_model safetensors initialization belongs to a long-running server outside the bounded measurement-harness contract",
    },
];

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
                | "deny"
                | "deprecated"
                | "doc"
                | "expect"
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

struct ParsedModuleSource {
    package: String,
    relative: String,
    parsed: StructuredSource,
}

impl ParsedModuleSource {
    fn key(&self) -> String {
        package_path_key(&self.package, &self.relative)
    }
}

fn parsed_package_module_closure(
    package: &ScannedPackage,
    root: &Path,
    test_cfg: bool,
) -> Result<Vec<ParsedModuleSource>, String> {
    let mut sources = parsed_module_closure(&package.manifest_dir, root, test_cfg)
        .map_err(|reason| format!("package {}: {reason}", package.identity))?;
    for source in &mut sources {
        source.package = package.identity.clone();
        source.parsed.context = source.key();
    }
    Ok(sources)
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
            Ok(ParsedModuleSource {
                package: ".".into(),
                relative,
                parsed,
            })
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
        Ok([
            "MetalEmbeddingGemma2State",
            "MetalErnie45State",
            "MetalQwen35State",
            "MetalForwardPass",
            "QwenModel",
        ]
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
    if sources.len() != 1
        || sources[0].package != contract.package
        || sources[0].relative != contract.path
    {
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
fn in_crate_raw_command_buffer_tests_are_explicit_and_guarded() {
    let packages = scanned_packages();
    let mut discovered = BTreeSet::new();
    let mut violations = Vec::new();
    for package in &packages {
        for root in package
            .target_roots(&["lib", "test"])
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            for source in parsed_package_module_closure(package, &root, true)
                .unwrap_or_else(|reason| panic!("{reason}"))
            {
                let relative = source.key();
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
    }

    let reviewed = IN_CRATE_COMMAND_BUFFER_TESTS
        .iter()
        .map(|(package, site)| package_path_key(package, site))
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

#[test]
fn rust_targets_have_one_gpu_lock_definition_and_path() {
    let mut packages = scanned_packages();
    packages.extend(
        scanned_packages_at(
            Path::new(env!("CARGO_MANIFEST_DIR")),
            &[LOCK_DEFINITION_PACKAGE],
        )
        .unwrap_or_else(|reason| panic!("{reason}")),
    );
    let definition = ["fn gpu", "_test_lock"].concat();
    let lock_path = ["/tmp/lion-metal", "-gpu-test.lock"].concat();
    let mut sources = Vec::new();
    for package in &packages {
        for path in rust_sources_under(&package.manifest_dir) {
            let relative = package.key(
                &package
                    .relative(&path)
                    .unwrap_or_else(|reason| panic!("{reason}")),
            );
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|reason| panic!("{relative}: {reason}"));
            sources.push((relative, source));
        }
    }
    let (definitions, paths) = lock_source_inventory(&sources, &definition, &lock_path)
        .unwrap_or_else(|reason| {
            panic!("receipt-reader exemption ({RECEIPT_READER_REASON}): {reason}")
        });

    let only_shared_module = BTreeSet::from([package_path_key(
        LOCK_DEFINITION_PACKAGE,
        "src/measurement.rs",
    )]);
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
    let packages = scanned_packages();
    let callers = [
        (
            "../models/qwen3-embedding",
            "src/metal.rs",
            "use lattice_metal_runtime::measurement::gpu_test_lock;",
        ),
        (
            "../models/qwen35",
            "src/forward/metal_qwen35/inner/tests.rs",
            "use lattice_metal_runtime::measurement::gpu_test_lock;",
        ),
        (
            ".",
            "tests/vision_s3b_vit_metal_gate_test.rs",
            "use lattice_inference::measurement::gpu_test_lock;",
        ),
        (".", "src/bin/bench_gdn_prefill_ab.rs", SHARED_LOCK_CALL),
    ];

    for (owner, relative, shared_reference) in callers {
        let package = measurement_lock_helpers::require_package(&packages, owner);
        let key = package.key(relative);
        let source = std::fs::read_to_string(package.manifest_dir.join(relative))
            .unwrap_or_else(|reason| panic!("{key}: {reason}"));
        assert!(
            source.contains(shared_reference),
            "{key} no longer references the one shared GPU lock"
        );
    }
}
