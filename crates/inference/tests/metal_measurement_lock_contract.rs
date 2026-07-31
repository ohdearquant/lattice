use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

const RAW_GPU_MARKERS: &[&str] = &["Device::system_default()", "new_command_buffer()"];
const SHARED_LOCK_CALL: &str = "lattice_inference::measurement::gpu_test_lock()";
const RAW_HARNESS_ENTRYPOINTS: &[(&str, &str)] = &[
    ("benches/decode_attn_bench.rs", "bench_flash_decode(c);"),
    ("benches/topk_readback.rs", "bench_topk_parity(c);"),
    ("examples/bench_concurrent.rs", "run()"),
    ("examples/bench_dispatch.rs", "Device::system_default()"),
    ("examples/bench_dispatch2.rs", "Device::system_default()"),
    ("examples/bench_mps_gemm.rs", "Device::system_default()"),
    ("examples/bench_simdgroup.rs", "Device::system_default()"),
    (
        "examples/profile_metal_decode.rs",
        "Device::system_default()",
    ),
];
const ADDITIONAL_GUARDED_ENTRYPOINTS: &[(&str, &str)] = &[
    ("examples/bench_embed_quality.rs", "run_bench();"),
    ("examples/bench_metal.rs", "MetalForwardPass::new("),
];
const REVIEWED_TOP_LEVEL_METAL_TARGETS: &[&str] = &[
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
    "src/bin/chat_metal.rs",
    "src/bin/dump_quarot_q4_golden.rs",
    "src/bin/eval_perplexity.rs",
    "src/bin/gramperf_profile.rs",
    "src/bin/lattice.rs",
    "src/bin/lattice/prune_score.rs",
    "src/bin/lattice_serve.rs",
    "src/bin/ppl_metal.rs",
];
const REVIEWED_NON_METAL_TOP_LEVEL_TARGETS: &[&str] = &[
    "benches/attention_dispatch_bench.rs",
    "benches/attn_opt_bench.rs",
    "benches/compute_attention_bench.rs",
    "benches/differential_attention_bench.rs",
    "benches/e2e_bench.rs",
    "benches/elementwise_bench.rs",
    "benches/elementwise_cpu_bench.rs",
    "benches/f16_convert_bench.rs",
    "benches/gated_attention_bench.rs",
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
    "src/bin/moe_admission_sim.rs",
    "src/bin/quantize_q4.rs",
    "src/bin/quantize_quarot.rs",
    "src/bin/qwen35_debug.rs",
    "src/bin/qwen35_generate.rs",
];
const IN_CRATE_COMMAND_BUFFER_TESTS: &[&str] = &[
    "test_gemv_decode_numerical",
    "test_gpu_argmax_parity_k1",
    "decode_attention_reference_fails_closed_on_nan_q",
    "gemv_q3_decode_matches_cpu_reference_across_shapes",
    "gemm_q3_tiled_matches_cpu_reference_at_tile_boundaries",
    "gemm_q3_tiled_differential_fails_closed_on_nan_scale",
    "gemv_q3_decode_mutation_sensitive_high_plane_bit",
    "lora_gemv_kernel_matches_cpu_reference",
    "load_adapter_and_dispatch_lora_if_active",
    "dispatch_matmul_q4_writes_all_rows",
];
const IN_CRATE_COMMAND_BUFFER_EXEMPTIONS: &[(&str, &str)] = &[(
    "test_gpu_argmax_parity_k1",
    "existing raw-dispatch test; migration is tracked with the remaining Metal lock work",
)];

const CONSTRUCTION_METHODS: &[&str] = &["new", "from_q4_dir"];
const LEGACY_CRITERION: &str =
    "legacy Criterion target; source-level locking needs separate benchmark evidence";
const LEGACY_EXAMPLE: &str =
    "legacy manually launched measurement example; lock migration is tracked separately";
const LONG_RUNNING: &str =
    "long-running process; a lifetime lock would starve measurements or exceed its bounded wait";
const CONSTRUCTION_EXEMPTIONS: &[(&str, &str)] = &[
    ("benches/cross_turn_prefix_cache_bench.rs", LEGACY_CRITERION),
    ("benches/lm_head_bench.rs", LEGACY_CRITERION),
    ("benches/metal_decode_bench.rs", LEGACY_CRITERION),
    ("benches/mtp_decode.rs", LEGACY_CRITERION),
    ("examples/bench_gdn_decode.rs", LEGACY_EXAMPLE),
    ("examples/bench_gdn_prefill_ab.rs", LEGACY_EXAMPLE),
    ("examples/bench_gdn_state.rs", LEGACY_EXAMPLE),
    ("examples/bench_persistent_state.rs", LEGACY_EXAMPLE),
    ("examples/bench_pruning.rs", LEGACY_EXAMPLE),
    ("examples/bench_q4_prefill.rs", LEGACY_EXAMPLE),
    ("examples/bench_q8_prefill.rs", LEGACY_EXAMPLE),
    ("examples/bench_quality.rs", LEGACY_EXAMPLE),
    ("examples/bench_stability.rs", LEGACY_EXAMPLE),
    ("examples/bench_suite.rs", LEGACY_EXAMPLE),
    ("examples/decode_profile.rs", LEGACY_EXAMPLE),
    ("examples/profile_metal.rs", LEGACY_EXAMPLE),
    ("src/bin/chat_metal.rs", LONG_RUNNING),
    ("src/bin/lattice.rs", LONG_RUNNING),
    ("src/bin/lattice/prune_score.rs", LONG_RUNNING),
    ("src/bin/lattice_serve.rs", LONG_RUNNING),
    (
        "tests/quarot_q4_composed_golden.rs",
        "opt-in real-model gate runs as a serialized step on an isolated CI runner",
    ),
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

fn constructions(source: &str) -> Vec<usize> {
    let mut sites = Vec::new();
    for method in CONSTRUCTION_METHODS {
        let marker = format!("MetalQwen35State::{method}(");
        sites.extend(source.match_indices(&marker).map(|(offset, _)| offset));
    }
    sites.sort_unstable();
    sites
}

fn top_level_metal_entrypoint_markers() -> Vec<String> {
    vec![
        "Device::system_default()".to_string(),
        "new_command_buffer()".to_string(),
        "MetalQwen35State".to_string(),
        "MetalForwardPass".to_string(),
        "QwenModel".to_string(),
    ]
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

struct GuardScope {
    protected: Range<usize>,
    binding: String,
}

fn leading_spaces(line: &str) -> usize {
    line.bytes().take_while(|byte| *byte == b' ').count()
}

fn rust_code_mask(source: &str) -> Vec<u8> {
    let bytes = source.as_bytes();
    let mut code = bytes.to_vec();
    let mut offset = 0;
    while offset < bytes.len() {
        let (start, end) = if bytes[offset..].starts_with(b"//") {
            let end = bytes[offset..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |length| offset + length);
            (offset, end)
        } else if bytes[offset..].starts_with(b"/*") {
            let mut end = offset + 2;
            let mut depth = 1;
            while end < bytes.len() && depth > 0 {
                if bytes[end..].starts_with(b"/*") {
                    depth += 1;
                    end += 2;
                } else if bytes[end..].starts_with(b"*/") {
                    depth -= 1;
                    end += 2;
                } else {
                    end += 1;
                }
            }
            (offset, end)
        } else if bytes[offset] == b'r' || bytes[offset..].starts_with(b"br") {
            let prefix = if bytes[offset] == b'b' { 2 } else { 1 };
            let mut quote = offset + prefix;
            while quote < bytes.len() && bytes[quote] == b'#' {
                quote += 1;
            }
            if quote >= bytes.len() || bytes[quote] != b'"' {
                offset += 1;
                continue;
            }
            let hashes = quote - offset - prefix;
            let mut end = quote + 1;
            while end < bytes.len() {
                if bytes[end] == b'"'
                    && bytes.get(end + 1..end + 1 + hashes) == Some(&bytes[offset + prefix..quote])
                {
                    end += 1 + hashes;
                    break;
                }
                end += 1;
            }
            (offset, end)
        } else if bytes[offset] == b'"' {
            let mut end = offset + 1;
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end += 2;
                } else if bytes[end] == b'"' {
                    end += 1;
                    break;
                } else {
                    end += 1;
                }
            }
            (offset, end.min(bytes.len()))
        } else {
            offset += 1;
            continue;
        };
        for byte in &mut code[start..end] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
        offset = end;
    }
    code
}

fn closing_scope(code: &[u8], start: usize, indentation: usize) -> Option<usize> {
    let closing_indentation = indentation.checked_sub(4)?;
    let mut offset = start;
    for line in code[start..].split_inclusive(|byte| *byte == b'\n') {
        let line = std::str::from_utf8(line).expect("masked Rust source remains UTF-8");
        if leading_spaces(line) == closing_indentation && line.trim() == "}" {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn guard_statement_is_present_in_metal_build(
    source: &str,
    code: &[u8],
    line_start: usize,
    indentation: usize,
) -> bool {
    let Some(previous_end) = code[..line_start]
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
    else {
        return true;
    };
    let previous_start = code[..previous_end]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |offset| offset + 1);
    let previous_line = &source[previous_start..=previous_end];
    if leading_spaces(previous_line) != indentation {
        return true;
    }
    let attribute = previous_line.trim();
    if attribute == "#[cfg(all(target_os = \"macos\", feature = \"metal-gpu\"))]" {
        return true;
    }
    !attribute.starts_with("#[") && !attribute.ends_with(']')
}

fn guard_scopes(source: &str, lock_call: &str) -> Vec<GuardScope> {
    let code = rust_code_mask(source);
    source
        .match_indices(lock_call)
        .filter_map(|(lock, _)| {
            if code[lock] == b' ' {
                return None;
            }
            let line_start = source[..lock].rfind('\n').map_or(0, |offset| offset + 1);
            let line_end = source[lock..]
                .find('\n')
                .map_or(source.len(), |offset| lock + offset);
            let line = &source[line_start..line_end];
            let indentation = leading_spaces(line);
            if !guard_statement_is_present_in_metal_build(source, &code, line_start, indentation) {
                return None;
            }
            let statement = line.trim().strip_prefix("let ")?;
            let (binding, expression) = statement.split_once(" = ")?;
            if binding == "_"
                || !binding
                    .bytes()
                    .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
                || expression != format!("{lock_call};")
            {
                return None;
            }
            let scope_end = closing_scope(&code, line_end + 1, indentation)?;
            Some(GuardScope {
                protected: line_end..scope_end,
                binding: binding.to_string(),
            })
        })
        .collect()
}

fn contains_identifier(source: &str, identifier: &str) -> bool {
    source.match_indices(identifier).any(|(offset, _)| {
        let before = source[..offset].bytes().next_back();
        let after = source[offset + identifier.len()..].bytes().next();
        let is_identifier = |byte: u8| byte == b'_' || byte.is_ascii_alphanumeric();
        !before.is_some_and(is_identifier) && !after.is_some_and(is_identifier)
    })
}

fn lock_held_through_scope(source: &str, lock_call: &str, work: usize) -> Result<(), String> {
    let Some(scope) = guard_scopes(source, lock_call)
        .into_iter()
        .find(|scope| scope.protected.contains(&work))
    else {
        return Err("no function-scope shared-lock binding encloses the Metal work".to_string());
    };
    if contains_identifier(&source[scope.protected.clone()], &scope.binding) {
        return Err(format!(
            "shared-lock binding `{}` is used before its scope ends and may be moved or dropped",
            scope.binding
        ));
    }
    Ok(())
}

fn lock_held_through_function(source: &str, lock_call: &str, work: usize) -> Result<(), String> {
    let function_end = source.rfind('\n').map_or(0, |line_break| line_break + 1);
    let Some(scope) = guard_scopes(source, lock_call)
        .into_iter()
        .find(|scope| scope.protected.contains(&work) && scope.protected.end == function_end)
    else {
        return Err(
            "no function-lifetime shared-lock binding encloses the Metal construction".to_string(),
        );
    };
    if contains_identifier(&source[scope.protected.clone()], &scope.binding) {
        return Err(format!(
            "function-lifetime shared-lock binding `{}` is used before the test ends and may be moved or dropped",
            scope.binding
        ));
    }
    Ok(())
}

fn assert_lock_held_across(source: &str, lock_call: &str, work: usize, context: &str) {
    if let Err(reason) = lock_held_through_scope(source, lock_call, work) {
        panic!("{context} does not hold the shared GPU lock across Metal work: {reason}");
    }
}

fn assert_guard_fixture_rejected(source: &str, context: &str) {
    let work = source
        .find("Device::system_default()")
        .expect("fixture contains Metal work");
    let result = std::panic::catch_unwind(|| {
        assert_lock_held_across(source, SHARED_LOCK_CALL, work, context);
    });
    assert!(result.is_err(), "source contract accepted {context}");
}

struct TestFunction<'a> {
    name: &'a str,
    source: &'a str,
}

fn test_functions(source: &str) -> Vec<TestFunction<'_>> {
    let code = rust_code_mask(source);
    let mut functions = Vec::new();
    let mut cursor = 0;
    while let Some(attribute) = source[cursor..].find("#[test]").map(|pos| cursor + pos) {
        let Some(function) = source[attribute..].find("fn ").map(|pos| attribute + pos) else {
            break;
        };
        let Some(open_brace) = source[function..].find('{').map(|pos| function + pos) else {
            break;
        };
        let line_start = source[..function]
            .rfind('\n')
            .map_or(0, |offset| offset + 1);
        let indentation = leading_spaces(&source[line_start..function]);
        let Some(close_line) = closing_scope(&code, open_brace + 1, indentation + 4) else {
            break;
        };
        let close_brace = close_line
            + source[close_line..]
                .find('}')
                .expect("closing scope line contains a brace");
        let name_start = function + "fn ".len();
        let Some(name_end) = source[name_start..].find('(').map(|pos| name_start + pos) else {
            break;
        };
        functions.push(TestFunction {
            name: &source[name_start..name_end],
            source: &source[attribute..close_brace + 1],
        });
        cursor = close_brace + 1;
    }
    functions
}

#[test]
fn raw_metal_measurement_harnesses_use_live_lock_bindings_across_the_entrypoint() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut actual = BTreeSet::new();

    for relative_dir in ["benches", "examples", "src/bin"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read measurement source");
            if !RAW_GPU_MARKERS.iter().any(|marker| source.contains(marker)) {
                continue;
            }

            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            actual.insert(relative.clone());

            let protected_entrypoint = RAW_HARNESS_ENTRYPOINTS
                .iter()
                .find_map(|(path, marker)| (*path == relative).then_some(*marker))
                .unwrap_or_else(|| panic!("raw Metal harness {relative} is not classified"));
            let work = source
                .find(protected_entrypoint)
                .unwrap_or_else(|| panic!("{relative} lost its protected Metal entrypoint"));
            assert_lock_held_across(&source, SHARED_LOCK_CALL, work, &relative);
        }
    }

    let expected = RAW_HARNESS_ENTRYPOINTS
        .iter()
        .map(|(path, _)| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "raw Metal harness inventory changed; classify every added or removed path explicitly"
    );
}

#[test]
fn top_level_metal_entrypoint_inventory_is_explicit_and_fail_closed() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut discovered = BTreeSet::new();
    let mut all_targets = BTreeSet::new();
    for relative_dir in ["benches", "examples", "src/bin"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read measurement source");
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            all_targets.insert(relative.clone());
            if top_level_metal_entrypoint_markers()
                .iter()
                .any(|marker| source.contains(marker.as_str()))
            {
                discovered.insert(relative);
            }
        }
    }

    let reviewed = REVIEWED_TOP_LEVEL_METAL_TARGETS
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered, reviewed,
        "top-level Metal entrypoint inventory changed; classify direct, helper-mediated, and \
         alternate-state-family paths explicitly"
    );
    let reviewed_non_metal = REVIEWED_NON_METAL_TOP_LEVEL_TARGETS
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
        let work = source
            .find(entrypoint)
            .unwrap_or_else(|| panic!("{relative} lost its reviewed Metal entrypoint"));
        assert_lock_held_across(&source, SHARED_LOCK_CALL, work, relative);
    }
}

#[test]
fn in_crate_raw_command_buffer_tests_are_explicit_and_guarded() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(manifest_dir.join("src/forward/metal_qwen35.rs"))
        .expect("read in-crate Metal tests");
    let exemptions = IN_CRATE_COMMAND_BUFFER_EXEMPTIONS
        .iter()
        .map(|(name, reason)| {
            assert!(
                !reason.trim().is_empty(),
                "{name} needs an exemption reason"
            );
            (*name, *reason)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut discovered = BTreeSet::new();
    let mut violations = Vec::new();
    for function in test_functions(&source) {
        let command_buffers = function
            .source
            .match_indices("new_command_buffer()")
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        if command_buffers.is_empty() {
            continue;
        }
        discovered.insert(function.name);
        if exemptions.contains_key(function.name) {
            continue;
        }
        for work in command_buffers {
            if let Err(reason) =
                lock_held_through_function(function.source, "gpu_test_lock()", work)
            {
                violations.push(format!("{}: {reason}", function.name));
            }
        }
    }

    let reviewed = IN_CRATE_COMMAND_BUFFER_TESTS
        .iter()
        .copied()
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
fn metal_qwen35_state_construction_tests_use_function_lifetime_lock_bindings() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exemptions = CONSTRUCTION_EXEMPTIONS
        .iter()
        .map(|(path, reason)| {
            assert!(
                !reason.trim().is_empty(),
                "construction exemption {path} needs a justification"
            );
            ((*path).to_string(), *reason)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        exemptions.len(),
        CONSTRUCTION_EXEMPTIONS.len(),
        "construction exemptions must not contain duplicate paths"
    );

    let mut discovered = BTreeSet::new();
    let mut classified = exemptions.keys().cloned().collect::<BTreeSet<_>>();
    let mut violations = Vec::new();
    for relative_dir in ["benches", "examples", "src/bin", "tests"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read construction source");
            let construction_sites = constructions(&source);
            if construction_sites.is_empty() {
                continue;
            }
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            discovered.insert(relative.clone());

            if exemptions.contains_key(&relative) {
                continue;
            }
            let failures = construction_sites
                .into_iter()
                .filter_map(|work| lock_held_through_scope(&source, SHARED_LOCK_CALL, work).err())
                .collect::<Vec<_>>();
            let protected_wrapper = RAW_HARNESS_ENTRYPOINTS
                .iter()
                .find_map(|(path, marker)| (*path == relative).then_some(*marker))
                .and_then(|marker| source.find(marker))
                .is_some_and(|work| {
                    lock_held_through_scope(&source, SHARED_LOCK_CALL, work).is_ok()
                });
            if failures.is_empty() || protected_wrapper {
                classified.insert(relative);
            } else {
                violations.push(format!("{relative}: {}", failures.join("; ")));
            }
        }
    }

    let in_crate_relative = "src/forward/metal_qwen35.rs";
    let in_crate_source =
        std::fs::read_to_string(manifest_dir.join(in_crate_relative)).expect("read in-crate tests");
    for function in test_functions(&in_crate_source) {
        let construction_sites = constructions(function.source);
        if construction_sites.is_empty() {
            continue;
        }
        let site = format!("{in_crate_relative}::{}", function.name);
        discovered.insert(site.clone());
        let failures = construction_sites
            .into_iter()
            .filter_map(|work| {
                lock_held_through_function(function.source, "gpu_test_lock()", work).err()
            })
            .collect::<Vec<_>>();
        if failures.is_empty() {
            classified.insert(site);
        } else {
            violations.push(format!("{site}: {}", failures.join("; ")));
        }
    }

    assert!(
        violations.is_empty(),
        "MetalQwen35State construction sites without a live shared-lock binding:\n{}",
        violations.join("\n")
    );

    assert_construction_inventory_classified(&discovered, &classified);
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
