use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const RAW_GPU_MARKERS: &[&str] = &["Device::system_default()", "new_command_buffer()"];
const SHARED_LOCK_CALL: &str = "lattice_inference::measurement::gpu_test_lock()";
const EXPECTED_RAW_HARNESSES: &[&str] = &[
    "benches/decode_attn_bench.rs",
    "benches/topk_readback.rs",
    "examples/bench_concurrent.rs",
    "examples/bench_dispatch.rs",
    "examples/bench_dispatch2.rs",
    "examples/bench_mps_gemm.rs",
    "examples/bench_simdgroup.rs",
    "examples/profile_metal_decode.rs",
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

fn first_construction(source: &str) -> Option<usize> {
    CONSTRUCTION_METHODS
        .iter()
        .filter_map(|method| source.find(&format!("MetalQwen35State::{method}(")))
        .min()
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

#[test]
fn raw_metal_measurement_harnesses_acquire_the_shared_lock_first() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut actual = BTreeSet::new();

    for relative_dir in ["benches", "examples", "src/bin"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read measurement source");
            let Some(first_gpu_operation) = RAW_GPU_MARKERS
                .iter()
                .filter_map(|marker| source.find(marker))
                .min()
            else {
                continue;
            };

            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            actual.insert(relative.clone());

            let lock = source.find(SHARED_LOCK_CALL).unwrap_or_else(|| {
                panic!("{relative} drives raw Metal work without the shared GPU lock")
            });
            assert!(
                lock < first_gpu_operation,
                "{relative} acquires the shared GPU lock after its first raw Metal operation"
            );
        }
    }

    let expected = EXPECTED_RAW_HARNESSES
        .iter()
        .map(|path| (*path).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "raw Metal harness inventory changed; classify every added or removed path explicitly"
    );
}

#[test]
fn metal_qwen35_state_construction_sites_acquire_the_shared_lock_first() {
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
    for relative_dir in ["benches", "examples", "src/bin", "tests"] {
        for path in rust_sources_under(&manifest_dir.join(relative_dir)) {
            let source = std::fs::read_to_string(&path).expect("read construction source");
            let Some(first_construction) = first_construction(&source) else {
                continue;
            };
            let relative = path
                .strip_prefix(manifest_dir)
                .expect("source under manifest directory")
                .to_string_lossy()
                .into_owned();
            discovered.insert(relative.clone());

            if source
                .find(SHARED_LOCK_CALL)
                .is_some_and(|lock| lock < first_construction)
            {
                classified.insert(relative);
            }
        }
    }

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
