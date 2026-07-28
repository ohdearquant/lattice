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
