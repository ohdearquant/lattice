use crate::quarot::plan::RotationTarget;

fn layer_prefix(idx: usize) -> String {
    format!("decoder.block.{idx}")
}

fn target() -> RotationTarget {
    RotationTarget::new(3, 8, 16, vec![0, 2], true, layer_prefix).unwrap()
}

#[test]
fn neutral_seed_bridge_preserves_legacy_fallback_without_inventing_index_seed() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("quantize_index.json"), r#"{"tensors":[]}"#).unwrap();
    let target = target();
    let missing =
        crate::__internal::read_quarot_seed_from_index(tmp.path(), &target, None).unwrap_err();
    assert!(missing.contains("no recoverable rotation seed"));
    let index_seed =
        crate::__internal::read_quarot_seed_from_index(tmp.path(), &target, Some(77)).unwrap();
    assert_eq!(index_seed, None);
    assert_eq!(index_seed.or(Some(77)), Some(77));
}

#[test]
fn neutral_seed_bridge_keeps_manifest_seed_ahead_of_legacy_seed() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"{"quarot_seed":42,"tensors":[]}"#,
    )
    .unwrap();
    let target = target();
    let index_seed =
        crate::__internal::read_quarot_seed_from_index(tmp.path(), &target, Some(77)).unwrap();
    assert_eq!(index_seed, Some(42));
    assert_eq!(index_seed.or(Some(77)), Some(42));
}

#[test]
fn neutral_seed_bridge_refuses_top_level_online_recipes_with_legacy_seed() {
    let tmp = tempfile::tempdir().unwrap();
    let target = target();
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"{"artifact_version":"v1-online-r3r4","quarot_seed":42,"online":{"version":"v0-residual","online_rotations":[],"asymmetric_tensor_names":[]},"tensors":[]}"#,
    ).unwrap();
    let err =
        crate::__internal::read_quarot_seed_from_index(tmp.path(), &target, Some(77)).unwrap_err();
    assert_eq!(
        err,
        format!(
            "{}: this runtime does not yet execute V1 online rotation recipes; artifact requires R3/R4 runtime support",
            tmp.path().join("quantize_index.json").display()
        )
    );
}

#[test]
fn neutral_seed_bridge_refuses_embedded_online_recipes_with_legacy_seed() {
    let tmp = tempfile::tempdir().unwrap();
    let target = target();
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"{"quarot_seed":42,"online":{"version":"v1-online-r3r4","online_rotations":[],"asymmetric_tensor_names":[]},"tensors":[]}"#,
    ).unwrap();
    let err =
        crate::__internal::read_quarot_seed_from_index(tmp.path(), &target, Some(77)).unwrap_err();
    assert_eq!(
        err,
        format!(
            "{}: this runtime does not yet execute V1 online rotation recipes; artifact requires R3/R4 runtime support",
            tmp.path().join("quantize_index.json").display()
        )
    );
}
