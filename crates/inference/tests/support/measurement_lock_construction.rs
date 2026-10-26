use super::*;
use std::collections::BTreeMap;

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
        let recorded_position = package_path_key(exemption.package, exemption.recorded_position);
        if current_position != recorded_position {
            self.moved_exemptions.push(format!(
                "{site}: recorded position `{recorded_position}` no longer matches current position \
                 `{current_position}`; confirm this exemption still describes the same work \
                 (this is not a locking violation)"
            ));
        }
        None
    }
}

pub(super) struct ConstructionInventory {
    pub(super) discovered: BTreeSet<String>,
    pub(super) classified: BTreeSet<String>,
    pub(super) used_exemptions: BTreeSet<String>,
    pub(super) expected_exemptions: BTreeSet<String>,
    pub(super) moved_exemptions: Vec<String>,
    pub(super) violations: Vec<String>,
}

pub(super) fn scan_constructions(
    packages: &[ScannedPackage],
    exemptions: &[ConstructionExemption],
) -> ConstructionInventory {
    let expected_exemption_count = exemptions.len();
    let exemptions = exemptions
        .iter()
        .map(|exemption| {
            assert!(
                !exemption.reason.trim().is_empty(),
                "construction exemption {} needs a justification",
                exemption.site
            );
            measurement_lock_helpers::require_package(packages, exemption.package);
            let (target, site) = exemption
                .site
                .split_once("=>")
                .expect("construction exemption must name its target and source site");
            (
                format!(
                    "{}=>{}",
                    package_path_key(exemption.package, target),
                    package_path_key(exemption.package, site)
                ),
                exemption,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        exemptions.len(),
        expected_exemption_count,
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
    for package in packages {
        for target in package
            .cargo_targets(&["bench", "example", "bin", "test", "lib"])
            .unwrap_or_else(|reason| panic!("{reason}"))
        {
            let target_relative = package
                .relative(&target.path)
                .unwrap_or_else(|reason| panic!("{reason}"));
            let sources = measurement_lock_helpers::construction_scan_sources(package, &target);
            let caller_owned = BENCH_HARNESSES.iter().find(|contract| {
                contract.package == package.identity
                    && contract.path == target_relative
                    && !contract.calls.is_empty()
            });
            if let Some(contract) = caller_owned {
                validate_bench_harness_sources(&sources, contract)
                    .unwrap_or_else(|reason| panic!("{}: {reason}", contract.key()));
            }
            for source in &sources {
                let site_prefix = package.key(&format!(
                    "{}:{}:{}",
                    target.kind, target.name, target_relative
                ));
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
    }

    ConstructionInventory {
        discovered: tally.discovered,
        classified: tally.classified,
        used_exemptions: tally.used_exemptions,
        expected_exemptions: exemptions.keys().cloned().collect(),
        moved_exemptions: tally.moved_exemptions,
        violations,
    }
}

#[test]
fn metal_qwen35_state_construction_tests_use_function_lifetime_lock_bindings() {
    let inventory = scan_constructions(&scanned_packages(), CONSTRUCTION_EXEMPTIONS);
    assert!(
        inventory.violations.is_empty(),
        "MetalQwen35State construction sites without a live shared-lock binding:\n{}",
        inventory.violations.join("\n")
    );
    assert!(
        inventory.moved_exemptions.is_empty(),
        "construction exemptions recorded a position that no longer matches (not a locking \
         violation — update recorded_position once the move is confirmed intentional):\n{}",
        inventory.moved_exemptions.join("\n")
    );

    assert_eq!(
        inventory.used_exemptions, inventory.expected_exemptions,
        "construction exemptions must identify current exact sites"
    );
    assert_construction_inventory_classified(&inventory.discovered, &inventory.classified);
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
