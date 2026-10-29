//! Staged target admission: selected owners run; unresolved work stays explicit.
//! Standalone contracts are declared and checked structurally, not executed
//! by this admission test. Linked modules and unselected work remain separate.

use super::*;
use std::collections::{BTreeMap, BTreeSet};
use syn::{Expr, Item};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Owner {
    Bench,
    Raw,
    Additional,
    State,
    Generate,
    Worker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Exemption {
    Backfill,
}

#[derive(Clone, Copy, Debug)]
enum Status {
    Selected(&'static [Owner]),
    Exempt(Exemption),
    Pending(&'static str),
}

#[derive(Clone, Copy, Debug)]
struct Record {
    package: &'static str,
    path: &'static str,
    standalone: Option<&'static str>,
    status: Status,
}

// These separate test binaries are inventoried, but this admission test does
// not execute them. The host must run them without skipped filters.
/// Targets that remain `Status::Pending`. #1527 stays open while this is nonzero.
const KNOWN_PENDING_TARGETS: usize = 17;

const STANDALONE_BINARIES: &[&str] = &[
    "bench_embedding_lock_contract",
    "bench_metal_work_lock_contract",
    "bench_suite_lock_contract",
    "gdn_profile_lock_contract",
    "gramperf_profile_lock_contract",
    "ppl_metal_lock_contract",
    "prefill_reset_lock_contract",
    "pruning_lock_contract",
    "quality_lock_contract",
    "rss_probe_lock_contract",
    "slopefit_lock_contract",
    "text_dump_lock_contract",
    "vision_dump_lock_contract",
    "vision_prefill_lock_contract",
];

const RECORDS: &[Record] = &[
    Record {
        package: ".",
        path: "benches/cross_turn_prefix_cache_bench.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench]),
    },
    Record {
        package: ".",
        path: "benches/decode_attn_bench.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench, Owner::Raw]),
    },
    Record {
        package: ".",
        path: "benches/lm_head_bench.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench]),
    },
    Record {
        package: ".",
        path: "benches/metal_decode_bench.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench]),
    },
    Record {
        package: ".",
        path: "benches/mtp_decode.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench]),
    },
    Record {
        package: ".",
        path: "benches/topk_readback.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Bench, Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_concurrent.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_decode_streaming.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Generate]),
    },
    Record {
        package: ".",
        path: "examples/bench_dispatch.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_dispatch2.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_embed_quality.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Additional]),
    },
    Record {
        package: ".",
        path: "examples/bench_embedding.rs",
        standalone: Some("bench_embedding_lock_contract"),
        status: Status::Pending(
            "standalone #2431 encode contract is not invoked by this admission",
        ),
    },
    Record {
        package: ".",
        path: "examples/bench_gdn_decode.rs",
        standalone: None,
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_gdn_prefill_ab.rs",
        standalone: None,
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_gdn_state.rs",
        standalone: Some("gdn_profile_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_gemm_shapes.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_gpu.rs",
        standalone: None,
        status: Status::Pending("WGPU-only arm requires shared measurement lock; fix handed"),
    },
    Record {
        package: ".",
        path: "examples/bench_metal.rs",
        standalone: Some("bench_metal_work_lock_contract"),
        status: Status::Selected(&[Owner::Additional]),
    },
    Record {
        package: ".",
        path: "examples/bench_metal_forward_allocation.rs",
        standalone: None,
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_mps_gemm.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_persistent_state.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Generate]),
    },
    Record {
        package: ".",
        path: "examples/bench_profile.rs",
        standalone: None,
        status: Status::Pending("Qwen encode owner handed, pending integration"),
    },
    Record {
        package: ".",
        path: "examples/bench_pruning.rs",
        standalone: Some("pruning_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_q4_prefill.rs",
        standalone: Some("prefill_reset_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_q8_prefill.rs",
        standalone: Some("prefill_reset_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/bench_quality.rs",
        standalone: Some("quality_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: "../serve",
        path: "examples/bench_serve_prepare.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Worker]),
    },
    Record {
        package: ".",
        path: "examples/bench_simdgroup.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/bench_stability.rs",
        standalone: None,
        status: Status::Pending("streaming owner handed, pending integration"),
    },
    Record {
        package: ".",
        path: "examples/bench_suite.rs",
        standalone: Some("bench_suite_lock_contract"),
        status: Status::Pending("standalone suite contract is not invoked by this admission"),
    },
    Record {
        package: ".",
        path: "examples/decode_profile.rs",
        standalone: Some("gdn_profile_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/dump_embeddinggemma2.rs",
        standalone: Some("text_dump_lock_contract"),
        status: Status::Pending(
            "standalone text callback contract is not invoked by this admission",
        ),
    },
    Record {
        package: ".",
        path: "examples/dump_embeddinggemma2_vision.rs",
        standalone: Some("vision_dump_lock_contract"),
        status: Status::Pending(
            "standalone #2438 vision callback contract is not invoked by this admission",
        ),
    },
    Record {
        package: ".",
        path: "examples/layer_sweep.rs",
        standalone: None,
        status: Status::Pending("source classification requires an explicit non-measurement proof"),
    },
    Record {
        package: ".",
        path: "examples/profile_metal.rs",
        standalone: None,
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "examples/profile_metal_decode.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    },
    Record {
        package: ".",
        path: "examples/rss_autorelease_probe.rs",
        standalone: Some("rss_probe_lock_contract"),
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "src/bin/backfill_qwen3.rs",
        standalone: None,
        status: Status::Exempt(Exemption::Backfill),
    },
    Record {
        package: ".",
        path: "src/bin/bench_decode_ab.rs",
        standalone: None,
        status: Status::Pending("chat-completion/reset work lacks owner; chat stack active"),
    },
    Record {
        package: ".",
        path: "src/bin/bench_decode_slopefit.rs",
        standalone: Some("slopefit_lock_contract"),
        status: Status::Selected(&[Owner::Generate]),
    },
    Record {
        package: ".",
        path: "src/bin/bench_gdn_prefill_ab.rs",
        standalone: None,
        status: Status::Pending("two chunk closures and reset sites lack owner"),
    },
    Record {
        package: ".",
        path: "src/bin/bench_logit_dump.rs",
        standalone: None,
        status: Status::Selected(&[Owner::State]),
    },
    Record {
        package: ".",
        path: "src/bin/bench_lora_mixture.rs",
        standalone: None,
        status: Status::Pending("adapter and generation work lacks owner; LoRA stack active"),
    },
    Record {
        package: ".",
        path: "src/bin/bench_vision_prefill_ab.rs",
        standalone: Some("vision_prefill_lock_contract"),
        status: Status::Pending(
            "standalone #2438 prefill closure contract is not invoked by this admission",
        ),
    },
    Record {
        package: ".",
        path: "src/bin/chat_metal.rs",
        standalone: None,
        status: Status::Pending("product-path exemption lacks target-wide validation"),
    },
    Record {
        package: ".",
        path: "src/bin/dump_quarot_q4_golden.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Generate]),
    },
    Record {
        package: ".",
        path: "src/bin/eval_perplexity.rs",
        standalone: None,
        status: Status::Pending("two guarded helper callers and LoRA/perplexity tail lack owner"),
    },
    Record {
        package: ".",
        path: "src/bin/gramperf_profile.rs",
        standalone: Some("gramperf_profile_lock_contract"),
        status: Status::Pending(
            "standalone #2431 generation/reset contract is not invoked by this admission",
        ),
    },
    Record {
        package: "../serve",
        path: "src/bin/lattice/main.rs",
        standalone: None,
        status: Status::Pending("serving product-path exemption lacks target-wide validation"),
    },
    Record {
        package: ".",
        path: "src/bin/ppl_metal.rs",
        standalone: Some("ppl_metal_lock_contract"),
        status: Status::Pending(
            "standalone #2431 perplexity contract is not invoked by this admission",
        ),
    },
];

fn key(record: &Record) -> String {
    package_path_key(record.package, record.path)
}

fn validate_records(
    discovered: &BTreeSet<String>,
    owners: &BTreeMap<String, BTreeSet<Owner>>,
    valid_exemptions: &BTreeSet<String>,
    records: &[Record],
) -> Result<Vec<String>, String> {
    let mut seen = BTreeSet::new();
    let mut pending = Vec::new();
    for record in records {
        let key = key(record);
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate target record: {key}"));
        }
        if !discovered.contains(&key) {
            return Err(format!("stale target record: {key}"));
        }
        let actual = owners.get(&key).cloned().unwrap_or_default();
        match record.status {
            Status::Selected(declared) => {
                let declared_set = declared.iter().copied().collect::<BTreeSet<_>>();
                if declared.is_empty()
                    || declared_set.len() != declared.len()
                    || declared_set != actual
                {
                    return Err(format!("selected validator set changed: {key}"));
                }
            }
            Status::Exempt(Exemption::Backfill) => {
                if record.package != "."
                    || record.path != "src/bin/backfill_qwen3.rs"
                    || !actual.is_empty()
                    || !valid_exemptions.contains(&key)
                {
                    return Err(format!("unvalidated exemption: {key}"));
                }
            }
            Status::Pending(reason) => {
                if reason.trim().is_empty() || !actual.is_empty() {
                    return Err(format!("invalid or stale unresolved record: {key}"));
                }
                pending.push(format!("{key}: {reason}"));
            }
        }
    }
    if &seen != discovered {
        return Err(format!(
            "unrecorded executable roots: {:?}",
            discovered.difference(&seen).collect::<Vec<_>>()
        ));
    }
    if owners.keys().any(|owner| !discovered.contains(owner)) {
        return Err("validator names an unrecognized executable root".into());
    }
    Ok(pending)
}

fn selected_owners() -> BTreeMap<String, BTreeSet<Owner>> {
    let mut owners = BTreeMap::<String, BTreeSet<Owner>>::new();
    let mut add = |package: &str, path: &str, owner: Owner| {
        if !(path.starts_with("benches/")
            || path.starts_with("examples/")
            || path.starts_with("src/bin/"))
        {
            return;
        }
        let key = package_path_key(package, path);
        assert!(
            owners.entry(key.clone()).or_default().insert(owner),
            "duplicate validator: {key}"
        );
    };
    for target in BENCH_HARNESSES {
        add(target.package, target.path, Owner::Bench);
    }
    for (package, path, _) in RAW_HARNESS_ENTRYPOINTS {
        add(package, path, Owner::Raw);
    }
    for (package, path, _) in ADDITIONAL_GUARDED_ENTRYPOINTS {
        add(package, path, Owner::Additional);
    }
    for (package, path) in TARGETS_WITH_STATE_DRIVEN_WORK {
        add(package, path, Owner::State);
    }
    for target in GENERATE_WORK_TARGETS {
        add(target.package, target.path, Owner::Generate);
    }
    add("../serve", "examples/bench_serve_prepare.rs", Owner::Worker);
    owners
}

fn validate_backfill_exemption(source: &str) -> bool {
    // These anchors and the construction exemption bind this classification
    // to the reviewed migration; other work classes remain outside its scope.
    source.contains("rusqlite::Connection::open(&db_path)")
        && source.contains("QwenModel::from_directory(&model_dir)")
        && source.contains("for (namespace, _entry_uuid, embedding_id, content) in &rows")
        && CONSTRUCTION_EXEMPTIONS.iter().any(|exemption| {
            exemption.package == "."
                && exemption.site == "bin:backfill_qwen3:src/bin/backfill_qwen3.rs=>src/bin/backfill_qwen3.rs::main::QwenModel::from_directory()#1"
                && exemption.reason.contains("long-running batch process")
        })
}

fn included_target_roots(source: &str) -> Result<BTreeSet<String>, String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut roots = BTreeSet::new();
    for item in &file.items {
        let Item::Const(item) = item else { continue };
        let Expr::Macro(expr) = item.expr.as_ref() else {
            continue;
        };
        if !expr.mac.path.is_ident("include_str") {
            continue;
        }
        let path = syn::parse2::<syn::LitStr>(expr.mac.tokens.clone())
            .map_err(|error| error.to_string())?
            .value();
        let Some(relative) = path.strip_prefix("../") else {
            continue;
        };
        if !(relative.starts_with("examples/") || relative.starts_with("src/bin/")) {
            continue;
        }
        if item.attrs.iter().any(|attr| !attr.path().is_ident("doc")) || !expr.attrs.is_empty() {
            return Err("conditional or attributed target include".into());
        }
        if !roots.insert(package_path_key(".", relative)) {
            return Err("duplicate target include".into());
        }
    }
    if roots.is_empty() {
        return Err("test binary has no direct target include".into());
    }
    Ok(roots)
}

fn validate_standalone_contracts(
    packages: &[ScannedPackage],
    records: &[Record],
) -> Result<(), String> {
    let inference = packages
        .iter()
        .find(|package| package.identity == ".")
        .ok_or("missing inference package")?;
    let test_roots = inference
        .target_roots(&["test"])?
        .into_iter()
        .map(|path| inference.relative(&path))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut declared = BTreeSet::new();
    for record in records {
        if let Some(binary) = record.standalone
            && (record.package != "."
                || !STANDALONE_BINARIES.contains(&binary)
                || !declared.insert((key(record), binary.to_owned())))
        {
            return Err(format!("invalid standalone declaration: {}", key(record)));
        }
    }
    let mut observed = BTreeSet::new();
    let mut binaries = BTreeSet::new();
    for binary in STANDALONE_BINARIES {
        if !binaries.insert(*binary) {
            return Err(format!("duplicate test binary: {binary}"));
        }
        let relative = format!("tests/{binary}.rs");
        if !test_roots.contains(&relative) {
            return Err(format!("missing Cargo test binary: {binary}"));
        }
        let source = std::fs::read_to_string(inference.manifest_dir.join(&relative))
            .map_err(|error| format!("{relative}: {error}"))?;
        for root in included_target_roots(&source)? {
            observed.insert((root, (*binary).to_owned()));
        }
    }
    if declared != observed {
        return Err(format!(
            "standalone target mapping changed: declared {declared:?}; observed {observed:?}"
        ));
    }
    Ok(())
}

#[test]
fn staged_target_admission_preserves_unresolved_gaps_and_does_not_close_1527() {
    // Execute the actual owner validators in this test, including when it is
    // selected by name. Mere membership in a table is not an exercised owner.
    measurement_lock_bench_harness::cargo_target_lexical_metal_marker_inventory_is_explicit();
    measurement_lock_bench_harness::raw_metal_measurement_harnesses_use_live_lock_bindings_across_the_entrypoint();
    measurement_lock_bench_harness::alternate_and_helper_mediated_entrypoints_hold_the_shared_lock(
    );
    measurement_lock_bench_harness::state_driven_metal_measurement_targets_use_live_lock_bindings();
    measurement_lock_generate::reviewed_metal_generate_targets_use_live_lock_bindings();
    measurement_lock_helpers::measurement_worker_lock_covers_factory_and_runtime();

    let discovered = TARGETS_WITH_RECOGNIZED_METAL_MARKERS
        .iter()
        .map(|(package, path)| package_path_key(package, path))
        .collect::<BTreeSet<_>>();
    let listed_count = TARGETS_WITH_RECOGNIZED_METAL_MARKERS.len();
    assert_eq!(
        discovered.len(),
        listed_count,
        "duplicate recognized target"
    );
    let packages = scanned_packages();
    validate_standalone_contracts(&packages, RECORDS)
        .unwrap_or_else(|error| panic!("standalone contract inventory: {error}"));
    let mut discovered = discovered;
    for package in &packages {
        for root in package
            .target_roots(&["bench", "example", "bin"])
            .unwrap_or_else(|e| panic!("{e}"))
        {
            let sources = parsed_package_module_closure(package, &root, false)
                .unwrap_or_else(|e| panic!("{e}"));
            if sources.iter().any(|source| {
                [SHARED_LOCK_SELECTOR, LOCAL_LOCK_SELECTOR]
                    .iter()
                    .any(|selector| {
                        source
                            .parsed
                            .has_selector(*selector)
                            .unwrap_or_else(|e| panic!("{e}"))
                    })
            }) {
                let relative = package.relative(&root).unwrap_or_else(|e| panic!("{e}"));
                discovered.insert(package.key(&relative));
            }
        }
    }
    let mut exemptions = BTreeSet::new();
    for record in RECORDS {
        let package = packages
            .iter()
            .find(|p| p.identity == record.package)
            .unwrap_or_else(|| panic!("missing package {}", record.package));
        let bytes = std::fs::read(package.manifest_dir.join(record.path))
            .unwrap_or_else(|error| panic!("{}: {error}", key(record)));
        if matches!(record.status, Status::Exempt(Exemption::Backfill))
            && let Ok(source) = std::str::from_utf8(&bytes)
            && validate_backfill_exemption(source)
        {
            exemptions.insert(key(record));
        }
    }
    let pending = validate_records(&discovered, &selected_owners(), &exemptions, RECORDS)
        .unwrap_or_else(|error| panic!("target admission: {error}"));
    eprintln!(
        "#1527 admission: {} targets lack a selected work owner; selected owners cover only their named work classes:",
        pending.len()
    );
    for gap in &pending {
        eprintln!("  {gap}");
    }
    assert_eq!(
        pending.len(),
        KNOWN_PENDING_TARGETS,
        "#1527 admission: the number of targets without a selected work owner changed; \
         update KNOWN_PENDING_TARGETS in the change that admits or adds a target"
    );
}

#[test]
fn standalone_target_includes_must_be_direct_and_unconditional() {
    let source = "const SOURCE: &str = include_str!(\"../examples/a.rs\");";
    assert_eq!(
        included_target_roots(source).unwrap(),
        BTreeSet::from([".:examples/a.rs".to_owned()])
    );
    assert!(included_target_roots("// include_str!(\"../examples/a.rs\")").is_err());
    assert!(included_target_roots(&format!("#[cfg(any())] {source}")).is_err());
    assert!(included_target_roots(&format!("{source} {source}")).is_err());
    assert_ne!(
        included_target_roots(source).unwrap(),
        BTreeSet::from([".:examples/b.rs".to_owned()])
    );
}

#[test]
fn admission_exemption_requires_the_reviewed_backfill_anchors() {
    let source = include_str!("../../src/bin/backfill_qwen3.rs");
    assert!(validate_backfill_exemption(source));
    let anchor = "rusqlite::Connection::open(&db_path)";
    assert!(source.contains(anchor));
    assert!(!validate_backfill_exemption(&source.replacen(
        anchor,
        "other(&db_path)",
        1
    )));
}

#[test]
fn admission_rejects_unrecorded_stale_duplicate_and_unexercised_owners() {
    const A: Record = Record {
        package: ".",
        path: "examples/a.rs",
        standalone: None,
        status: Status::Selected(&[Owner::Raw]),
    };
    const B: Record = Record {
        package: ".",
        path: "src/bin/b.rs",
        standalone: None,
        status: Status::Pending("work not yet checked"),
    };
    let a = key(&A);
    let b = key(&B);
    let discovered = BTreeSet::from([a.clone(), b.clone()]);
    let owners = BTreeMap::from([(a.clone(), BTreeSet::from([Owner::Raw]))]);
    let exemptions = BTreeSet::new();
    assert_eq!(
        validate_records(&discovered, &owners, &exemptions, &[A, B])
            .unwrap()
            .len(),
        1
    );
    assert!(validate_records(&discovered, &owners, &exemptions, &[A]).is_err());
    assert!(validate_records(&discovered, &owners, &exemptions, &[A, B, B]).is_err());
    assert!(validate_records(&BTreeSet::from([a.clone()]), &owners, &exemptions, &[A, B]).is_err());
    assert!(validate_records(&discovered, &BTreeMap::new(), &exemptions, &[A, B]).is_err());
    let mut newly_owned = owners.clone();
    newly_owned.insert(b.clone(), BTreeSet::from([Owner::State]));
    assert!(validate_records(&discovered, &newly_owned, &exemptions, &[A, B]).is_err());
    let fake_exemption = Record {
        status: Status::Exempt(Exemption::Backfill),
        ..B
    };
    assert!(validate_records(&discovered, &owners, &exemptions, &[A, fake_exemption]).is_err());
    let blank = Record {
        status: Status::Pending(" "),
        ..B
    };
    assert!(validate_records(&discovered, &owners, &exemptions, &[A, blank]).is_err());
    let overlap = Record {
        status: Status::Selected(&[Owner::Raw, Owner::State]),
        ..A
    };
    let mut overlapping_owners = owners.clone();
    overlapping_owners.insert(a, BTreeSet::from([Owner::Raw, Owner::State]));
    assert!(validate_records(&discovered, &overlapping_owners, &exemptions, &[overlap, B]).is_ok());
}
