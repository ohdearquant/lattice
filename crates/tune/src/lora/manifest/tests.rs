use super::*;

fn sample_entry(id: &str, status: AdapterStatus) -> ManifestEntry {
    ManifestEntry {
        id: id.to_string(),
        name: format!("{id} test adapter"),
        owner: "team-test".to_string(),
        uri: format!("adapters/{id}.safetensors"),
        integrity_sha256: "abc123def456".to_string(),
        base_model_rev: "main".to_string(),
        tokenizer_rev: "main".to_string(),
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dtype: "f32".to_string(),
        status,
    }
}

#[test]
fn roundtrip_json() {
    let mut manifest = LoraManifest::new();
    manifest
        .adapters
        .push(sample_entry("adapter-1", AdapterStatus::Approved));
    manifest
        .adapters
        .push(sample_entry("adapter-2", AdapterStatus::Quarantined));

    let json = manifest.to_json().unwrap();
    let parsed = LoraManifest::from_json(&json).unwrap();
    assert_eq!(parsed.version, 1);
    assert_eq!(parsed.adapters.len(), 2);
    assert_eq!(parsed.adapters[0].id, "adapter-1");
    assert_eq!(parsed.adapters[0].status, AdapterStatus::Approved);
    assert_eq!(parsed.adapters[1].id, "adapter-2");
    assert_eq!(parsed.adapters[1].status, AdapterStatus::Quarantined);
    assert_eq!(parsed.adapters[0].rank, 8);
    assert!((parsed.adapters[0].alpha - 16.0).abs() < 1e-6);
    assert_eq!(parsed.adapters[0].owner, "team-test");
    assert_eq!(parsed.adapters[0].dtype, "f32");
}

#[test]
fn rejects_missing_required_field() {
    // `id` is omitted — serde must reject this. `owner`/`dtype` are
    // filled in (even though they're also required) so `id` remains the
    // *sole* deliberately-missing field, matching this test's name.
    let json = r#"{
            "version": 1,
            "adapters": [{
                "name": "test", "owner": "team-test", "uri": "test.safetensors",
                "integrity_sha256": "abc",
                "base_model_rev": "none", "tokenizer_rev": "none",
                "rank": 8, "alpha": 16.0, "target_modules": ["q_proj"],
                "dtype": "f32", "status": "approved"
            }]
        }"#;
    assert!(LoraManifest::from_json(json).is_err());
}

#[test]
fn status_snake_case_roundtrip() {
    let approved = serde_json::to_string(&AdapterStatus::Approved).unwrap();
    assert_eq!(approved, "\"approved\"");

    let quarantined = serde_json::to_string(&AdapterStatus::Quarantined).unwrap();
    assert_eq!(quarantined, "\"quarantined\"");

    let revoked = serde_json::to_string(&AdapterStatus::Revoked).unwrap();
    assert_eq!(revoked, "\"revoked\"");

    let back: AdapterStatus = serde_json::from_str("\"approved\"").unwrap();
    assert_eq!(back, AdapterStatus::Approved);

    let back: AdapterStatus = serde_json::from_str("\"revoked\"").unwrap();
    assert_eq!(back, AdapterStatus::Revoked);
}

#[test]
fn status_as_str_matches_serde_wire_format() {
    for (status, expected) in [
        (AdapterStatus::Approved, "approved"),
        (AdapterStatus::Quarantined, "quarantined"),
        (AdapterStatus::Revoked, "revoked"),
    ] {
        assert_eq!(status.as_str(), expected);
        // `as_str()` must never drift from the serde wire format, since
        // callers (e.g. the safetensors header writer) use it as a
        // stand-in for JSON serialization.
        assert_eq!(
            serde_json::to_string(&status).unwrap(),
            format!("\"{expected}\"")
        );
    }
}

#[test]
fn rejects_unknown_status() {
    // `owner`/`dtype` are filled in so the failure is unambiguously
    // about the bad `status` value, not an incidental missing field.
    let json = r#"{
            "id": "x", "name": "n", "owner": "team-test", "uri": "u",
            "integrity_sha256": "h",
            "base_model_rev": "none", "tokenizer_rev": "none",
            "rank": 4, "alpha": 4.0,
            "target_modules": ["q_proj"],
            "dtype": "f32",
            "status": "provisional"
        }"#;
    assert!(serde_json::from_str::<ManifestEntry>(json).is_err());
}

#[test]
fn load_save_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("manifest.json");
    let mut manifest = LoraManifest::new();
    manifest
        .adapters
        .push(sample_entry("rev-adapter", AdapterStatus::Revoked));
    manifest.save(&path).unwrap();

    let loaded = LoraManifest::load(&path).unwrap();
    assert_eq!(loaded.adapters.len(), 1);
    assert_eq!(loaded.adapters[0].status, AdapterStatus::Revoked);
    assert_eq!(loaded.adapters[0].id, "rev-adapter");
    assert_eq!(loaded.adapters[0].base_model_rev, "main");
    assert_eq!(loaded.adapters[0].tokenizer_rev, "main");
    assert_eq!(loaded.adapters[0].owner, "team-test");
    assert_eq!(loaded.adapters[0].dtype, "f32");
}

#[test]
fn default_creates_empty_v1() {
    let m = LoraManifest::default();
    assert_eq!(m.version, 1);
    assert!(m.adapters.is_empty());
}

/// Mutation-sensitive: an over-cap manifest is rejected before parsing.
/// Removing the size guard in `load` lets the read proceed, and the file
/// (all NUL bytes) then fails to parse with a different error — so this
/// assertion on "exceeds maximum" fails when the guard is absent.
#[test]
fn load_rejects_oversized_manifest() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big_manifest.json");

    // Produce a file whose stat length is MAX_MANIFEST_SIZE + 1 without
    // writing 64 MiB: seek past the cap and write one byte. On every major
    // filesystem this yields a sparse file (one real disk sector) whose
    // reported length crosses the cap, exercising the guard cheaply.
    {
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(MAX_MANIFEST_SIZE)).unwrap();
        f.write_all(b"x").unwrap();
    }

    let err = LoraManifest::load(&path).unwrap_err().to_string();
    assert!(
        err.contains("exceeds maximum"),
        "expected the size guard to fire; got: {err}"
    );
}
