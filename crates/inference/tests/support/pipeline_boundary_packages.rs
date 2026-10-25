use super::*;

pub(super) struct BoundaryPopulation<'a> {
    pub(super) origins: &'a [(&'a str, &'a str)],
    pub(super) frontends: &'a [(&'a str, &'a str)],
    pub(super) providers: &'a [(&'a str, &'a str, &'a [&'a str])],
    pub(super) known_positive: &'a [(&'a str, &'a str)],
}

const PROVIDER_FILES: &[&str] = &[
    "bert.rs",
    "bert/embedding.rs",
    "gemma.rs",
    "gemma/worker.rs",
    "qwen.rs",
    "qwen/adapters.rs",
    "qwen/embedding.rs",
    "qwen/lattice_worker.rs",
    "qwen/vision.rs",
    "qwen/worker.rs",
];

pub(super) const PRODUCTION_POPULATION: BoundaryPopulation<'_> = BoundaryPopulation {
    origins: &[(".", "src/model"), (".", "src/forward/metal_*")],
    frontends: &[
        ("../serve", "src/serve/mod.rs"),
        ("../serve", "src/serving_worker.rs"),
    ],
    providers: &[("../serve", "src/serving_provider/providers", PROVIDER_FILES)],
    known_positive: &[
        ("../serve", "src/serve/mod.rs"),
        ("../serve", "src/serving_worker.rs"),
        ("../serve", "src/bin/lattice_serve/main.rs"),
    ],
};

pub(super) struct PackageBoundaryReport {
    pub(super) targets: Vec<(String, String, String)>,
    pub(super) modules_read: BTreeSet<String>,
    pub(super) crossings: BTreeSet<Crossing>,
    pub(super) unresolved: Vec<String>,
}

fn package_by_identity<'a>(
    packages: &'a [ScannedPackage],
    identity: &str,
) -> Result<&'a ScannedPackage, String> {
    packages
        .iter()
        .find(|package| package.identity == identity)
        .ok_or_else(|| format!("population names unscanned package {identity}"))
}

fn package_context(
    package: &ScannedPackage,
    libraries: &BTreeMap<String, String>,
) -> Result<CrossingCrateContext, String> {
    let output = std::process::Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--manifest-path",
        ])
        .arg(package.manifest_dir.join("Cargo.toml"))
        .output()
        .map_err(|reason| format!("package {}: cargo metadata: {reason}", package.identity))?;
    if !output.status.success() {
        return Err(format!(
            "package {}: cargo metadata failed: {}",
            package.identity,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|reason| format!("package {}: metadata JSON: {reason}", package.identity))?;
    let manifest = package.manifest_dir.join("Cargo.toml");
    let mut matching = Vec::new();
    for candidate in metadata["packages"]
        .as_array()
        .ok_or("metadata has no packages")?
    {
        let path = candidate["manifest_path"]
            .as_str()
            .ok_or("metadata package has no manifest")?;
        let path = std::fs::canonicalize(path).map_err(|reason| reason.to_string())?;
        if path == manifest {
            matching.push(candidate);
        }
    }
    if matching.len() != 1 {
        return Err(format!(
            "package {}: metadata selected {} manifests",
            package.identity,
            matching.len()
        ));
    }
    let mut crate_name_to_namespace = libraries.clone();
    let mut external_crate_names = BTreeSet::new();
    for dependency in matching[0]["dependencies"]
        .as_array()
        .ok_or("metadata package has no dependencies")?
    {
        let name = dependency["name"]
            .as_str()
            .ok_or("metadata dependency has no name")?
            .replace('-', "_");
        let alias = match &dependency["rename"] {
            serde_json::Value::Null => name.clone(),
            serde_json::Value::String(rename) if !rename.is_empty() => rename.replace('-', "_"),
            _ => return Err("metadata dependency has an invalid rename".into()),
        };
        if let Some(namespace) = libraries.get(&name) {
            if let Some(previous) = crate_name_to_namespace.insert(alias.clone(), namespace.clone())
                && previous != *namespace
            {
                return Err(format!(
                    "package {}: dependency namespace collision for {alias}",
                    package.identity
                ));
            }
        } else {
            external_crate_names.insert(alias);
        }
    }
    Ok(CrossingCrateContext {
        crate_name_to_namespace,
        external_crate_names,
    })
}

pub(super) fn collect_pipeline_boundary(
    packages: &[ScannedPackage],
    population: &BoundaryPopulation<'_>,
) -> Result<PackageBoundaryReport, String> {
    if packages.is_empty() {
        return Err("pipeline scanned package list is empty".into());
    }
    let identities = packages
        .iter()
        .map(|package| &package.identity)
        .collect::<BTreeSet<_>>();
    if identities.len() != packages.len() {
        return Err("pipeline scanned package identities are not unique".into());
    }
    let mut libraries = BTreeMap::new();
    let mut targets = BTreeMap::new();
    let mut frontend_roots: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut report_targets = Vec::new();
    let mut bin_count = 0;
    for package in packages {
        let package_targets = package.cargo_targets(&["lib", "bin"])?;
        let info = read_crate_manifest_info(&package.manifest_dir);
        if package_targets.iter().any(|target| target.kind == "lib") {
            let name = info
                .package_name
                .ok_or_else(|| format!("package {} has no library name", package.identity))?;
            if libraries
                .insert(name.clone(), package.key("crate"))
                .is_some()
            {
                return Err(format!("scanned packages repeat library name {name}"));
            }
        }
        let roots = frontend_roots.entry(package.identity.clone()).or_default();
        for target in &package_targets {
            let key = package.key(&package.relative(&target.path)?);
            report_targets.push((package.key(&target.name), target.kind.clone(), key));
            if target.kind == "bin" {
                roots.push(target.path.clone());
                bin_count += 1;
            }
        }
        targets.insert(package.identity.clone(), package_targets);
    }
    if bin_count == 0 {
        return Err("an empty bin-target set is a failure, not a clean run".into());
    }
    for (identity, relative) in population.frontends {
        let package = package_by_identity(packages, identity)?;
        frontend_roots
            .get_mut(*identity)
            .expect("scanned package initialized")
            .push(package.manifest_dir.join(relative));
    }
    for (identity, relative, files) in population.providers {
        let package = package_by_identity(packages, identity)?;
        let providers = package.manifest_dir.join(relative);
        let roots = files
            .iter()
            .map(|path| providers.join(path))
            .collect::<Vec<_>>();
        assert_provider_roots_complete(&providers, &roots);
        frontend_roots
            .get_mut(*identity)
            .expect("scanned package initialized")
            .extend(roots);
    }

    let mut contexts = BTreeMap::new();
    let mut walked = BTreeMap::new();
    let mut extra = Vec::new();
    let mut frontend_files = BTreeSet::new();
    let mut file_packages = BTreeMap::new();
    let mut namespace_packages = BTreeMap::new();
    for package in packages {
        contexts.insert(
            package.identity.clone(),
            package_context(package, &libraries)?,
        );
        let package_targets = &targets[&package.identity];
        let known_targets = package_targets
            .iter()
            .map(|target| target.path.clone())
            .collect::<BTreeSet<_>>();
        let mut namespaces = BTreeMap::new();
        for target in package_targets {
            let namespace = if target.kind == "lib" {
                package.key("crate")
            } else {
                package.key(&format!("bin::{}", target.name))
            };
            namespaces.insert(target.path.clone(), namespace);
        }
        let mut walker = Walker::new(&package.manifest_dir, false)?;
        let frontends = &frontend_roots[&package.identity];
        let index_roots = package_targets
            .iter()
            .filter(|target| target.kind == "lib")
            .map(|target| &target.path);
        for root in index_roots.chain(frontends.iter()) {
            let key = package.key(&package.relative(root)?);
            let canonical = std::fs::canonicalize(root)
                .map_err(|reason| format!("{key}: could not resolve root: {reason}"))?;
            if walker.files.contains_key(&canonical) {
                continue;
            }
            let namespace = namespaces
                .get(&canonical)
                .cloned()
                .unwrap_or_else(|| package.key("crate"));
            walker
                .walk_root(
                    root,
                    &namespace,
                    inferred_module_path(&canonical, &known_targets),
                )
                .map_err(|reason| format!("package {}: {reason}", package.identity))?;
        }
        for root in frontends {
            let key = package.key(&package.relative(root)?);
            let canonical = std::fs::canonicalize(root).map_err(|reason| reason.to_string())?;
            let mounted = walker
                .files
                .get(&canonical)
                .ok_or_else(|| format!("{key}: frontend root not discovered"))?;
            for (path, file) in &walker.files {
                if file.namespace == mounted.namespace
                    && file.module_path.starts_with(&mounted.module_path)
                {
                    frontend_files.insert(path.clone());
                }
            }
        }
        for (path, file) in &walker.files {
            if file_packages.insert(path.clone(), package).is_some() {
                return Err(format!(
                    "{}: source belongs to multiple scanned packages",
                    package.key(&package.relative(path)?)
                ));
            }
            namespace_packages.insert(file.namespace.clone(), package.identity.clone());
        }
        for (_, file) in &walker.extra_mounts {
            namespace_packages.insert(file.namespace.clone(), package.identity.clone());
        }
        walked.extend(walker.files);
        extra.extend(walker.extra_mounts);
    }
    if frontend_files.is_empty() {
        return Err(
            "no frontend files were discovered: an empty root set is a failure, not a clean run"
                .into(),
        );
    }
    let indices = build_index_with_contexts(&walked, &extra, |namespace| {
        &contexts[&namespace_packages[namespace]]
    });
    let mut origin_dirs = Vec::new();
    let mut origin_globs = Vec::new();
    for (identity, relative) in population.origins {
        let package = package_by_identity(packages, identity)?;
        let path = package.manifest_dir.join(relative.trim_end_matches('*'));
        if relative.ends_with('*') {
            let parent = path.parent().ok_or("origin glob has no parent")?;
            let parent = std::fs::canonicalize(parent)
                .map_err(|reason| format!("{}: {reason}", package.key(relative)))?;
            origin_globs.push(parent.join(path.file_name().ok_or("origin glob has no prefix")?));
        } else {
            origin_dirs.push(
                std::fs::canonicalize(path)
                    .map_err(|reason| format!("{}: {reason}", package.key(relative)))?,
            );
        }
    }
    if !population.origins.is_empty()
        && !walked
            .keys()
            .any(|path| is_origin_file(path, &origin_dirs, &origin_globs))
    {
        return Err("origin directories classified ZERO discovered files".into());
    }
    let mut report = PackageBoundaryReport {
        targets: report_targets,
        modules_read: BTreeSet::new(),
        crossings: BTreeSet::new(),
        unresolved: Vec::new(),
    };
    for path in frontend_files {
        let package = file_packages[&path];
        let file = &walked[&path];
        let key = package.key(&package.relative(&path)?);
        report.modules_read.insert(key.clone());
        for reason in [
            unclassifiable_item_macro(&file.file),
            unclassifiable_include(&file.file, false),
        ]
        .into_iter()
        .flatten()
        {
            report.unresolved.push(format!("{key}: {reason}"));
        }
        let resolver = Resolver {
            ctx: &contexts[&package.identity],
            indices: &indices,
        };
        let local_item_names = collect_local_item_names(&file.file);
        let mut scanner = CrossingScanner {
            resolver: &resolver,
            home_namespace: file.namespace.clone(),
            home_module_path: file.module_path.clone(),
            relative_path: key,
            origin_dir_abs: &origin_dirs,
            origin_glob_abs: &origin_globs,
            crossings: &mut report.crossings,
            unresolved: &mut report.unresolved,
            current_self_ty: Vec::new(),
            generic_scopes: Vec::new(),
            assoc_type_scopes: Vec::new(),
            local_item_names: &local_item_names,
        };
        scanner.visit_file(&file.file);
    }
    for (identity, relative) in population.known_positive {
        let package = package_by_identity(packages, identity)?;
        if !report.modules_read.contains(&package.key(relative)) {
            return Err(format!(
                "{}: known-positive root missing from modules_read",
                package.key(relative)
            ));
        }
    }
    Ok(report)
}

pub(super) fn analyze_boundary(
    crate_dir: &Path,
    origin_dirs: &[&str],
    frontend_roots: &[PathBuf],
    index_roots: &[PathBuf],
    test_cfg: bool,
) -> Result<BoundaryReport, String> {
    let manifest_info = read_crate_manifest_info(crate_dir);
    let mut crate_name_to_namespace = BTreeMap::new();

    let lib_targets = cargo_targets(crate_dir, &["lib"]);
    let bin_targets = cargo_targets(crate_dir, &["bin"]);
    let mut known_target_paths: BTreeSet<PathBuf> = BTreeSet::new();
    let mut target_namespace: BTreeMap<PathBuf, String> = BTreeMap::new();
    let mut report_targets = Vec::new();

    let is_real_crate = lib_targets.is_ok() || bin_targets.is_ok();

    if let Ok(targets) = &lib_targets {
        for target in targets {
            known_target_paths.insert(target.path.clone());
            target_namespace.insert(target.path.clone(), "crate".to_string());
            if let Some(name) = &manifest_info.package_name {
                crate_name_to_namespace.insert(name.clone(), "crate".to_string());
            }
        }
    }
    if let Ok(targets) = &bin_targets {
        for target in targets {
            known_target_paths.insert(target.path.clone());
            target_namespace.insert(target.path.clone(), format!("bin::{}", target.name));
        }
    }

    let ctx = CrossingCrateContext {
        crate_name_to_namespace,
        external_crate_names: manifest_info.dependency_names,
    };

    let mut walker = Walker::new(crate_dir, test_cfg)?;

    for root in index_roots.iter().chain(frontend_roots.iter()) {
        let canonical = std::fs::canonicalize(root)
            .map_err(|reason| format!("could not resolve root {}: {reason}", root.display()))?;
        if walker.files.contains_key(&canonical) {
            continue; // already discovered via an earlier root in this loop
        }
        let namespace = if !is_real_crate {
            "self".to_string()
        } else if let Some(ns) = target_namespace.get(&canonical) {
            ns.clone()
        } else {
            // Not itself a Cargo target: fold it into "crate" if it lives
            // under the same manifest dir as the lib target (the serve/
            // case), else give it a fresh, root-specific namespace.
            "crate".to_string()
        };
        let module_path = inferred_module_path(&canonical, &known_target_paths);
        walker.walk_root(root, &namespace, module_path.clone())?;
    }

    let indices = build_index_with_paths(&walker.files, &walker.extra_mounts, &ctx);
    let resolver = Resolver {
        ctx: &ctx,
        indices: &indices,
    };

    // A `filter_map` here would fail OPEN: a moved, renamed, or mistyped
    // origin directory silently drops out of the set rather than erroring,
    // leaving an empty (or partial) origin set that produces a report
    // reading like a completed migration -- zero crossings, no unresolved
    // entries, looking exactly like success. An origin directory that
    // cannot be found is a completeness failure of THIS CHECK, not an
    // empty contribution to it, so every entry is a hard error naming the
    // exact directory that did not resolve. Skipped entirely when
    // `origin_dirs` is itself empty: several fixtures deliberately pass
    // `&[]` to isolate a DIFFERENT concern (frontend-root discovery,
    // unreadable-source handling) from origin classification, and a
    // caller declaring no origin directories at all is not claiming
    // anything about how many of them got classified.
    let mut origin_dir_abs: Vec<PathBuf> = Vec::new();
    let mut origin_glob_abs: Vec<PathBuf> = Vec::new();
    if !origin_dirs.is_empty() {
        for d in origin_dirs.iter().filter(|d| !d.ends_with('*')) {
            let joined = crate_dir.join(d);
            let canonical = std::fs::canonicalize(&joined).map_err(|reason| {
                format!(
                    "origin directory {} did not resolve: {reason}",
                    joined.display()
                )
            })?;
            origin_dir_abs.push(canonical);
        }
        for d in origin_dirs.iter().filter(|d| d.ends_with('*')) {
            let prefix = d.trim_end_matches('*');
            let parent = crate_dir.join(prefix);
            let parent_dir = parent.parent().ok_or_else(|| {
                format!(
                    "origin glob {d} (joined: {}) has no parent directory",
                    parent.display()
                )
            })?;
            let parent_canonical = std::fs::canonicalize(parent_dir).map_err(|reason| {
                format!(
                    "origin glob {d}'s parent directory {} did not resolve: {reason}",
                    parent_dir.display()
                )
            })?;
            let file_prefix = Path::new(prefix)
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("origin glob {d} has no usable file-name prefix"))?;
            origin_glob_abs.push(parent_canonical.join(file_prefix));
        }
        // Known-positive on the ORIGIN side, matching the one the roots
        // side already has (`bin_targets` asserted non-empty below): an
        // origin set that resolved to real directories but happens to
        // classify nothing at all is exactly as vacuous as an empty set,
        // and reads exactly like a completed migration. This lives here,
        // in `analyze_boundary` itself, so every fixture inherits the
        // same guarantee its production caller gets -- a fixture whose
        // `model/` directory silently stopped classifying anything would
        // otherwise pass by reporting zero crossings, indistinguishable
        // from a fixture correctly demonstrating no crossing exists.
        let origin_item_count = walker
            .files
            .keys()
            .filter(|path| is_origin_file(path, &origin_dir_abs, &origin_glob_abs))
            .count();
        if origin_item_count == 0 {
            return Err(format!(
                "origin directories {origin_dirs:?} resolved to {} absolute dir(s) and {} \
                 glob prefix(es), but classified ZERO discovered files as origin -- an empty \
                 origin set makes every crossing classification in this run vacuous",
                origin_dir_abs.len(),
                origin_glob_abs.len()
            ));
        }
    }

    // Frontend membership: for each frontend root, every walked file in
    // the SAME namespace whose module path starts with this root's own
    // module path.
    let mut frontend_files: BTreeSet<PathBuf> = BTreeSet::new();
    for root in frontend_roots {
        let canonical = std::fs::canonicalize(root)
            .map_err(|reason| format!("could not resolve root {}: {reason}", root.display()))?;
        let Some(walked_root) = walker.files.get(&canonical) else {
            return Err(format!(
                "frontend root {} was not discovered by its own walk",
                root.display()
            ));
        };
        let root_namespace = walked_root.namespace.clone();
        let root_module_path = walked_root.module_path.clone();
        for (path, walked_file) in &walker.files {
            if walked_file.namespace == root_namespace
                && walked_file.module_path.starts_with(&root_module_path)
            {
                frontend_files.insert(path.clone());
            }
        }
    }

    let mut modules_read: Vec<PathBuf> = frontend_files.iter().cloned().collect();
    modules_read.sort();

    if modules_read.is_empty() {
        return Err(
            "no frontend files were discovered: an empty root set is a failure, not a clean run"
                .to_string(),
        );
    }

    let mut crossings = BTreeSet::new();
    let mut unresolved = Vec::new();

    for path in &modules_read {
        let walked_file = walker.files.get(path).expect("path came from walker.files");
        if let Some(reason) = unclassifiable_item_macro(&walked_file.file) {
            unresolved.push(format!(
                "{}: {reason}",
                path.strip_prefix(crate_dir).unwrap_or(path).display()
            ));
        }
        if let Some(reason) = unclassifiable_include(&walked_file.file, test_cfg) {
            unresolved.push(format!(
                "{}: {reason}",
                path.strip_prefix(crate_dir).unwrap_or(path).display()
            ));
        }
        let relative_path = path
            .strip_prefix(crate_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let local_item_names = collect_local_item_names(&walked_file.file);
        let mut scanner = CrossingScanner {
            resolver: &resolver,
            home_namespace: walked_file.namespace.clone(),
            home_module_path: walked_file.module_path.clone(),
            relative_path,
            origin_dir_abs: &origin_dir_abs,
            origin_glob_abs: &origin_glob_abs,
            crossings: &mut crossings,
            unresolved: &mut unresolved,
            current_self_ty: Vec::new(),
            generic_scopes: Vec::new(),
            assoc_type_scopes: Vec::new(),
            local_item_names: &local_item_names,
        };
        scanner.visit_file(&walked_file.file);
    }

    if let Ok(targets) = lib_targets {
        report_targets.extend(targets);
    }
    if let Ok(targets) = bin_targets {
        report_targets.extend(targets);
    }

    Ok(BoundaryReport {
        targets: report_targets,
        modules_read,
        crossings,
        unresolved,
    })
}

/// One sorted, tab-separated `relative_path<TAB>resolved_item` line per
/// unique crossing (kind is not part of the ratchet: two different
/// reference KINDS reaching the same item from the same file are one
/// population entry, not two).
pub(super) fn baseline_lines(report: &PackageBoundaryReport) -> BTreeSet<(String, String)> {
    report
        .crossings
        .iter()
        .map(|c| (c.relative_path.clone(), c.resolved_path.clone()))
        .collect()
}

pub(super) fn format_baseline(lines: &BTreeSet<(String, String)>) -> String {
    let mut out = String::new();
    for (path, item) in lines {
        out.push_str(path);
        out.push('\t');
        out.push_str(item);
        out.push('\n');
    }
    out
}

pub(super) fn parse_baseline(text: &str) -> BTreeSet<(String, String)> {
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.splitn(2, '\t');
            let path = parts.next().unwrap_or_default().to_string();
            let item = parts.next().unwrap_or_default().to_string();
            if path.contains(':') {
                (path, item)
            } else {
                (package_path_key(".", &path), package_path_key(".", &item))
            }
        })
        .collect()
}
