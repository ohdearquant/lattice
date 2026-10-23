use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

// Keep the historical exclusions fixed so each new extraction is covered before
// anybody remembers to update a second crate list. An unrelated new workspace
// member requires an explicit classification decision here.
pub fn is_extracted(name: &str) -> bool {
    !matches!(
        name,
        "lattice-inference"
            | "lattice-embed"
            | "lattice-tune"
            | "lattice-fann"
            | "lattice-transport"
    )
}

pub fn read_workspace_metadata() -> &'static Value {
    static METADATA: OnceLock<Value> = OnceLock::new();
    METADATA.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let output = Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
                "--locked",
                "--manifest-path",
            ])
            .arg(&manifest)
            .output()
            .expect("run cargo metadata for workspace contracts");
        assert!(
            output.status.success(),
            "cargo metadata failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("parse cargo metadata JSON")
    })
}

pub fn workspace_packages(metadata: &Value) -> Result<BTreeMap<String, &Value>, String> {
    if metadata.get("version").and_then(Value::as_u64) != Some(1) {
        return Err("cargo metadata format version must be 1".to_owned());
    }
    let packages = metadata
        .get("packages")
        .and_then(Value::as_array)
        .ok_or("cargo metadata packages must be an array")?;
    let members = metadata
        .get("workspace_members")
        .and_then(Value::as_array)
        .ok_or("cargo metadata workspace_members must be an array")?;
    if members.is_empty() {
        return Err("cargo metadata workspace_members is empty".to_owned());
    }
    let mut by_id = BTreeMap::new();
    for package in packages {
        let id = package
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or("cargo metadata package has no nonempty id")?;
        if by_id.insert(id, package).is_some() {
            return Err(format!("duplicate cargo metadata package id: {id}"));
        }
    }
    let mut seen = BTreeSet::new();
    let mut workspace = BTreeMap::new();
    for member in members {
        let id = member
            .as_str()
            .ok_or("cargo metadata workspace member must be a package id")?;
        if !seen.insert(id) {
            return Err(format!("duplicate cargo metadata workspace member: {id}"));
        }
        let package = *by_id
            .get(id)
            .ok_or_else(|| format!("workspace member {id} has no package record"))?;
        let name = package
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| format!("workspace member {id} has no nonempty name"))?;
        if workspace.insert(name.to_owned(), package).is_some() {
            return Err(format!("duplicate workspace package name: {name}"));
        }
    }
    for required in [
        "lattice-inference",
        "lattice-inference-core",
        "lattice-tokenizer",
        "lattice-grammar",
    ] {
        if !workspace.contains_key(required) {
            return Err(format!("workspace coverage control missing: {required}"));
        }
    }
    Ok(workspace)
}
