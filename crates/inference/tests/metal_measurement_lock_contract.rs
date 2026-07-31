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
            let statement = line.trim().strip_prefix("let ")?;
            let (binding, expression) = statement.split_once(" = ")?;
            if !binding
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
                || expression != format!("{lock_call};")
            {
                return None;
            }
            let scope_end = closing_scope(&code, line_end + 1, leading_spaces(line))?;
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
fn guard_lifetime_rejects_an_immediate_drop_before_metal_work() {
    let source = r#"
fn measurement() {
    let gpu_guard = lattice_inference::measurement::gpu_test_lock();
    drop(gpu_guard);
    let _device = Device::system_default();
}
"#;
    let work = source
        .find("Device::system_default()")
        .expect("fixture contains Metal work");

    let result = std::panic::catch_unwind(|| {
        assert_lock_held_across(source, SHARED_LOCK_CALL, work, "immediate-drop fixture");
    });
    assert!(
        result.is_err(),
        "source contract accepted a guard dropped before Metal work"
    );
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
