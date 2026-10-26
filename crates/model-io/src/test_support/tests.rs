use super::*;

#[test]
fn unset_refuses() {
    let err = resolve_checkpoint_dir("UNUSED", |_| None, |_| true).unwrap_err();
    assert!(err.contains("is unset"), "{err}");
}

#[test]
fn relative_refuses() {
    let err = resolve_checkpoint_dir("VAR", |_| Some("relative/checkpoint".to_string()), |_| true)
        .unwrap_err();
    assert!(err.contains("is relative"), "{err}");
}

#[test]
fn missing_refuses() {
    let err = resolve_checkpoint_dir(
        "VAR",
        |_| Some("/definitely/absolute/checkpoint".to_string()),
        |_| false,
    )
    .unwrap_err();
    assert!(err.contains("does not exist"), "{err}");
}

/// The must-pass arm: without it, the three refusal controls above would
/// all pass unconditionally against a helper that panicked (or, here,
/// returned `Err`) regardless of input.
#[test]
fn present_absolute_resolves() {
    let dir = resolve_checkpoint_dir(
        "VAR",
        |_| Some("/definitely/absolute/checkpoint".to_string()),
        |_| true,
    )
    .unwrap();
    assert_eq!(dir, PathBuf::from("/definitely/absolute/checkpoint"));
}
