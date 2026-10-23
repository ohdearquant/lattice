//! ADR-103 D7 checks declarations, including inactive target and optional edges.
//! A resolved default-feature graph would omit dependencies that can violate the
//! layering contract under another feature or target configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

#[path = "support/workspace_metadata.rs"]
mod workspace_metadata;

type Graph = BTreeMap<String, BTreeSet<String>>;

fn string_field<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{context}: {field} must be a nonempty string"))
}

fn nullable_string<'a>(
    value: &'a Value,
    field: &str,
    context: &str,
) -> Result<Option<&'a str>, String> {
    match value.get(field) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value)),
        _ => Err(format!(
            "{context}: {field} must be null or a nonempty string"
        )),
    }
}

fn workspace_graph(metadata: &Value) -> Result<Graph, String> {
    let packages = workspace_metadata::workspace_packages(metadata)?;
    let mut directories = BTreeMap::new();
    let mut graph = Graph::new();
    for (name, package) in &packages {
        let manifest = Path::new(string_field(package, "manifest_path", name)?);
        if !manifest.is_absolute() || manifest.file_name() != Some("Cargo.toml".as_ref()) {
            return Err(format!(
                "{name}: manifest_path must be an absolute Cargo.toml path"
            ));
        }
        let directory = manifest
            .parent()
            .ok_or_else(|| format!("{name}: manifest_path has no parent"))?;
        if directories.insert(directory, name.as_str()).is_some() {
            return Err(format!("{name}: duplicate workspace manifest directory"));
        }
        graph.insert(name.clone(), BTreeSet::new());
    }

    for (name, package) in &packages {
        let dependencies = package
            .get("dependencies")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{name}: dependencies must be an array"))?;
        for (index, dependency) in dependencies.iter().enumerate() {
            let context = format!("{name} dependency {index}");
            let target = string_field(dependency, "name", &context)?;
            match dependency.get("kind") {
                Some(Value::Null) => {}
                Some(Value::String(kind)) if kind == "build" || kind == "dev" => {}
                _ => return Err(format!("{context}: kind must be null, build, or dev")),
            }
            if dependency
                .get("optional")
                .and_then(Value::as_bool)
                .is_none()
            {
                return Err(format!("{context}: optional must be a boolean"));
            }
            nullable_string(dependency, "rename", &context)?;
            nullable_string(dependency, "target", &context)?;
            let source = nullable_string(dependency, "source", &context)?;
            let edges = graph
                .get_mut(name)
                .expect("every workspace package has a graph node");
            if source.is_some() {
                if dependency.get("path").is_some() {
                    return Err(format!(
                        "{context}: nonlocal dependency must not declare path"
                    ));
                }
                // Publishing the facade does not make an upward dependency legal.
                if target == "lattice-inference" {
                    edges.insert(target.to_owned());
                }
                continue;
            }

            let path = Path::new(string_field(dependency, "path", &context)?);
            if !path.is_absolute() {
                return Err(format!("{context}: path must be absolute"));
            }
            let package_name = directories.get(path).ok_or_else(|| {
                format!("{context}: local dependency path is outside the workspace")
            })?;
            if *package_name != target {
                return Err(format!(
                    "{context}: dependency name {target} disagrees with workspace package {package_name}"
                ));
            }
            edges.insert(target.to_owned());
        }
    }
    Ok(graph)
}

fn path_to_facade(
    graph: &Graph,
    node: &str,
    visited: &mut BTreeSet<String>,
    path: &mut Vec<String>,
) -> bool {
    if !visited.insert(node.to_owned()) {
        return false;
    }
    path.push(node.to_owned());
    if node == "lattice-inference" {
        return true;
    }
    for next in &graph[node] {
        if path_to_facade(graph, next, visited, path) {
            return true;
        }
    }
    path.pop();
    false
}

fn check_acyclic(
    graph: &Graph,
    node: &str,
    completed: &mut BTreeSet<String>,
    active: &mut Vec<String>,
) -> Result<(), String> {
    if let Some(start) = active.iter().position(|entry| entry == node) {
        let mut cycle = active[start..].to_vec();
        cycle.push(node.to_owned());
        return Err(format!(
            "workspace dependency cycle: {}",
            cycle.join(" -> ")
        ));
    }
    if completed.contains(node) {
        return Ok(());
    }
    active.push(node.to_owned());
    for next in &graph[node] {
        check_acyclic(graph, next, completed, active)?;
    }
    active.pop();
    completed.insert(node.to_owned());
    Ok(())
}

fn check_dependency_direction(metadata: &Value) -> Result<(), String> {
    let graph = workspace_graph(metadata)?;
    for name in graph
        .keys()
        .filter(|name| workspace_metadata::is_extracted(name))
    {
        let mut path = Vec::new();
        if path_to_facade(&graph, name, &mut BTreeSet::new(), &mut path) {
            return Err(format!(
                "extracted crate reaches lattice-inference: {}",
                path.join(" -> ")
            ));
        }
    }
    let mut completed = BTreeSet::new();
    for name in graph.keys() {
        check_acyclic(&graph, name, &mut completed, &mut Vec::new())?;
    }
    Ok(())
}

fn dependency(name: &str, kind: Option<&str>) -> Value {
    json!({
        "name": name,
        "kind": kind,
        "rename": null,
        "optional": false,
        "target": null,
        "source": null,
        "path": format!("/workspace/{name}"),
    })
}

fn add_package(metadata: &mut Value, name: &str) {
    let id = format!("path+file:///workspace/{name}#{name}@0.14.0");
    metadata["packages"].as_array_mut().unwrap().push(json!({
        "name": name,
        "id": id,
        "manifest_path": format!("/workspace/{name}/Cargo.toml"),
        "dependencies": [],
    }));
    metadata["workspace_members"]
        .as_array_mut()
        .unwrap()
        .push(json!(id));
}

fn fixture() -> Value {
    let mut metadata = json!({"version": 1, "packages": [], "workspace_members": []});
    for name in [
        "lattice-inference",
        "lattice-inference-core",
        "lattice-tokenizer",
        "lattice-grammar",
        "lattice-embed",
        "lattice-tune",
        "lattice-fann",
        "lattice-transport",
    ] {
        add_package(&mut metadata, name);
    }
    for target in [
        "lattice-inference-core",
        "lattice-tokenizer",
        "lattice-grammar",
    ] {
        add_dependency(&mut metadata, "lattice-inference", dependency(target, None));
    }
    for name in ["lattice-tokenizer", "lattice-grammar"] {
        add_dependency(
            &mut metadata,
            name,
            dependency("lattice-inference-core", None),
        );
    }
    metadata
}

fn package_mut<'a>(metadata: &'a mut Value, name: &str) -> &'a mut Value {
    metadata["packages"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|package| package["name"] == name)
        .unwrap()
}

fn add_dependency(metadata: &mut Value, name: &str, dependency: Value) {
    package_mut(metadata, name)["dependencies"]
        .as_array_mut()
        .unwrap()
        .push(dependency);
}

#[test]
fn workspace_dependency_direction_matches_adr_103() {
    let metadata = workspace_metadata::read_workspace_metadata();
    let graph =
        workspace_graph(metadata).expect("workspace metadata must describe a complete graph");
    for target in [
        "lattice-inference-core",
        "lattice-tokenizer",
        "lattice-grammar",
    ] {
        assert!(
            graph["lattice-inference"].contains(target),
            "missing positive control edge: {target}"
        );
    }
    check_dependency_direction(metadata).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn legal_extracted_and_legacy_dependencies_pass() {
    let mut metadata = fixture();
    add_dependency(
        &mut metadata,
        "lattice-grammar",
        dependency("lattice-tokenizer", Some("dev")),
    );
    for name in ["lattice-embed", "lattice-tune"] {
        let mut edge = dependency("lattice-inference", None);
        edge["optional"] = json!(true);
        add_dependency(&mut metadata, name, edge);
    }
    assert_eq!(check_dependency_direction(&metadata), Ok(()));
}

fn assert_direct_backedge(kind: Option<&str>) {
    let mut metadata = fixture();
    add_dependency(
        &mut metadata,
        "lattice-tokenizer",
        dependency("lattice-inference", kind),
    );
    assert_eq!(
        check_dependency_direction(&metadata),
        Err(
            "extracted crate reaches lattice-inference: lattice-tokenizer -> lattice-inference"
                .into()
        )
    );
}

#[test]
fn normal_backedge_fails() {
    assert_direct_backedge(None);
}

#[test]
fn build_backedge_fails() {
    assert_direct_backedge(Some("build"));
}

#[test]
fn dev_backedge_fails() {
    assert_direct_backedge(Some("dev"));
}

#[test]
fn renamed_optional_target_backedge_fails() {
    let mut metadata = fixture();
    let mut edge = dependency("lattice-inference", Some("build"));
    edge["rename"] = json!("engine_alias");
    edge["optional"] = json!(true);
    edge["target"] = json!("cfg(target_os = \"none\")");
    add_dependency(&mut metadata, "lattice-tokenizer", edge);
    assert_eq!(
        check_dependency_direction(&metadata),
        Err(
            "extracted crate reaches lattice-inference: lattice-tokenizer -> lattice-inference"
                .into()
        )
    );
}

#[test]
fn future_crate_transitive_backedge_through_legacy_crate_fails() {
    let mut metadata = fixture();
    add_package(&mut metadata, "lattice-weights");
    add_dependency(
        &mut metadata,
        "lattice-weights",
        dependency("lattice-embed", Some("dev")),
    );
    add_dependency(
        &mut metadata,
        "lattice-embed",
        dependency("lattice-inference", None),
    );
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("extracted crate reaches lattice-inference: lattice-weights -> lattice-embed -> lattice-inference".into())
    );
}

#[test]
fn extracted_crate_cycle_fails() {
    let mut metadata = fixture();
    add_dependency(
        &mut metadata,
        "lattice-grammar",
        dependency("lattice-tokenizer", Some("dev")),
    );
    add_dependency(
        &mut metadata,
        "lattice-tokenizer",
        dependency("lattice-grammar", Some("build")),
    );
    assert_eq!(
        check_dependency_direction(&metadata),
        Err(
            "workspace dependency cycle: lattice-grammar -> lattice-tokenizer -> lattice-grammar"
                .into()
        )
    );
}

#[test]
fn legacy_crate_cycle_fails() {
    let mut metadata = fixture();
    add_dependency(
        &mut metadata,
        "lattice-embed",
        dependency("lattice-tune", None),
    );
    add_dependency(
        &mut metadata,
        "lattice-tune",
        dependency("lattice-embed", Some("dev")),
    );
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("workspace dependency cycle: lattice-embed -> lattice-tune -> lattice-embed".into())
    );
}

#[test]
fn self_cycle_fails() {
    let mut metadata = fixture();
    add_dependency(
        &mut metadata,
        "lattice-inference-core",
        dependency("lattice-inference-core", None),
    );
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("workspace dependency cycle: lattice-inference-core -> lattice-inference-core".into())
    );
}

#[test]
fn registry_facade_backedge_fails() {
    let mut metadata = fixture();
    let mut edge = dependency("lattice-inference", None);
    edge["source"] = json!("registry+https://github.com/rust-lang/crates.io-index");
    edge.as_object_mut().unwrap().remove("path");
    add_dependency(&mut metadata, "lattice-tokenizer", edge);
    assert_eq!(
        check_dependency_direction(&metadata),
        Err(
            "extracted crate reaches lattice-inference: lattice-tokenizer -> lattice-inference"
                .into()
        )
    );
}

#[test]
fn external_registry_dependencies_are_not_workspace_nodes() {
    let mut metadata = fixture();
    for name in ["serde", "lattice-tokenizer"] {
        let mut edge = dependency(name, None);
        edge["source"] = json!("registry+https://github.com/rust-lang/crates.io-index");
        edge.as_object_mut().unwrap().remove("path");
        add_dependency(&mut metadata, "lattice-tokenizer", edge);
    }
    assert_eq!(check_dependency_direction(&metadata), Ok(()));
}

#[test]
fn malformed_dependency_fields_fail() {
    for (field, value, expected) in [
        ("name", json!(false), "name must be a nonempty string"),
        ("kind", json!("normal"), "kind must be null, build, or dev"),
        ("optional", json!(null), "optional must be a boolean"),
        (
            "rename",
            json!(false),
            "rename must be null or a nonempty string",
        ),
        (
            "target",
            json!([]),
            "target must be null or a nonempty string",
        ),
        (
            "source",
            json!({}),
            "source must be null or a nonempty string",
        ),
        ("path", json!(null), "path must be a nonempty string"),
        ("path", json!("../inference"), "path must be absolute"),
    ] {
        let mut metadata = fixture();
        let mut edge = dependency("lattice-inference", None);
        edge[field] = value;
        add_dependency(&mut metadata, "lattice-transport", edge);
        assert_eq!(
            check_dependency_direction(&metadata),
            Err(format!("lattice-transport dependency 0: {expected}"))
        );
    }
}

#[test]
fn missing_dependency_fields_fail() {
    for (field, expected) in [
        ("name", "name must be a nonempty string"),
        ("kind", "kind must be null, build, or dev"),
        ("optional", "optional must be a boolean"),
        ("rename", "rename must be null or a nonempty string"),
        ("target", "target must be null or a nonempty string"),
        ("source", "source must be null or a nonempty string"),
        ("path", "path must be a nonempty string"),
    ] {
        let mut metadata = fixture();
        let mut edge = dependency("lattice-inference", None);
        edge.as_object_mut().unwrap().remove(field);
        add_dependency(&mut metadata, "lattice-transport", edge);
        assert_eq!(
            check_dependency_direction(&metadata),
            Err(format!("lattice-transport dependency 0: {expected}"))
        );
    }
}

#[test]
fn registry_dependency_with_local_path_fails() {
    let mut metadata = fixture();
    let mut edge = dependency("lattice-inference", None);
    edge["source"] = json!("registry+https://github.com/rust-lang/crates.io-index");
    add_dependency(&mut metadata, "lattice-tokenizer", edge);
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("lattice-tokenizer dependency 1: nonlocal dependency must not declare path".into())
    );
}

#[test]
fn unknown_local_path_fails() {
    let mut metadata = fixture();
    let mut edge = dependency("lattice-inference", None);
    edge["path"] = json!("/outside/inference");
    add_dependency(&mut metadata, "lattice-tokenizer", edge);
    assert_eq!(
        check_dependency_direction(&metadata),
        Err(
            "lattice-tokenizer dependency 1: local dependency path is outside the workspace".into()
        )
    );
}

#[test]
fn name_path_disagreement_fails() {
    let mut metadata = fixture();
    let mut edge = dependency("innocent_alias", None);
    edge["path"] = json!("/workspace/lattice-inference");
    add_dependency(&mut metadata, "lattice-tokenizer", edge);
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("lattice-tokenizer dependency 1: dependency name innocent_alias disagrees with workspace package lattice-inference".into())
    );
}

#[test]
fn missing_dependencies_fail() {
    let mut metadata = fixture();
    package_mut(&mut metadata, "lattice-grammar")
        .as_object_mut()
        .unwrap()
        .remove("dependencies");
    assert_eq!(
        check_dependency_direction(&metadata),
        Err("lattice-grammar: dependencies must be an array".into())
    );
}

#[test]
fn malformed_or_empty_metadata_fails() {
    for metadata in [
        Value::Null,
        json!({"version": 2, "packages": [], "workspace_members": []}),
        json!({"version": 1, "packages": [], "workspace_members": []}),
        json!({"version": 1, "packages": [], "workspace_members": ["missing"]}),
    ] {
        assert!(check_dependency_direction(&metadata).is_err());
    }
}
