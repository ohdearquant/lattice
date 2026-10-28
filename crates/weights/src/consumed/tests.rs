use super::*;
use crate::SafetensorsFile;

#[test]
fn successful_open_reports_same_header_size_and_digest_path() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    let header = b"{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    let mut original = (header.len() as u64).to_le_bytes().to_vec();
    original.extend_from_slice(header);
    original.extend_from_slice(&1f32.to_le_bytes());
    std::fs::write(&path, &original).unwrap();
    for mode in [WeightDigestMode::Full, WeightDigestMode::Header] {
        let mut requests = Vec::new();
        let (parsed, report) =
            SafetensorsFile::open_with_report(&path, "weight", mode, &mut |request| {
                requests.push(request)
            })
            .unwrap();
        assert_eq!(parsed.tensor_shape("x"), Some(&[1][..]));
        assert_eq!(report.header_len(), header.len() as u64);
        assert_eq!(report.header_bytes(), header);
        assert_eq!(report.file_size(), original.len() as u64);
        let mut recipe = (header.len() as u64).to_le_bytes().to_vec();
        recipe.extend_from_slice(header);
        let content: [u8; 32] = Sha256::digest(&original).into();
        if mode == WeightDigestMode::Full {
            recipe.extend_from_slice(&content);
        }
        assert_eq!(
            report.recipe_digest(),
            <[u8; 32]>::from(Sha256::digest(&recipe))
        );
        assert_eq!(
            report.content_digest(),
            (mode == WeightDigestMode::Full).then_some(content)
        );
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.kind == DigestKind::SafetensorsContent)
                .count(),
            usize::from(mode == WeightDigestMode::Full)
        );
        assert!(
            requests
                .iter()
                .filter(|r| r.kind == DigestKind::SafetensorsContent)
                .all(|r| r.bytes == original.len() as u64)
        );
        assert_eq!(requests.last().unwrap().bytes, recipe.len() as u64);
    }
}

#[test]
fn malformed_open_emits_no_successful_digest_requests() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.safetensors");
    std::fs::write(&path, b"bad").unwrap();
    let mut requests = Vec::new();
    assert!(
        SafetensorsFile::open_with_report(&path, "weight", WeightDigestMode::Full, &mut |r| {
            requests.push(r)
        })
        .is_err()
    );
    assert!(requests.is_empty());
}
