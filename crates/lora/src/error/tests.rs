use super::LoraError;
use std::error::Error as _;

#[test]
fn message_has_no_category_prefix() {
    let e = LoraError::Format("tensor 'x' has unsupported dtype".into());
    assert_eq!(e.message(), "tensor 'x' has unsupported dtype");
    assert_eq!(
        e.to_string(),
        "format error: tensor 'x' has unsupported dtype"
    );
}

#[test]
fn io_keeps_its_kind_and_source() {
    let e = LoraError::from(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "too big",
    ));
    match &e {
        LoraError::Io(inner) => assert_eq!(inner.kind(), std::io::ErrorKind::InvalidData),
        other => panic!("expected Io, got {other:?}"),
    }
    assert!(e.source().is_some());
    assert_eq!(e.message(), "too big");
}
