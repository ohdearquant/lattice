use super::*;

#[test]
fn canonical_golden_all_fields_and_types() {
    let digest = |start: u8| std::array::from_fn(|index| start + index as u8);
    let descriptor = EmbeddingSpaceDescriptor {
        mode: IdentityMode::default(),
        output_revision: "r1".into(),
        producer_family: "bert".into(),
        artifacts: vec![
            Artifact {
                role: ArtifactRole::Safetensors,
                logical_identifier: "w".into(),
                file_size: 64,
                recipe_digest: digest(224),
            },
            Artifact {
                role: ArtifactRole::Tokenizer,
                logical_identifier: "b".into(),
                file_size: 5,
                recipe_digest: digest(192),
            },
            Artifact {
                role: ArtifactRole::Config,
                logical_identifier: "a".into(),
                file_size: 3,
                recipe_digest: digest(160),
            },
            Artifact {
                role: ArtifactRole::Safetensors,
                logical_identifier: "v".into(),
                file_size: 63,
                recipe_digest: digest(0),
            },
        ],
        policy: vec![
            PolicyEntry {
                key: "g".into(),
                value: PolicyValue::F64(1.5),
            },
            PolicyEntry {
                key: "c".into(),
                value: PolicyValue::Signed(-2),
            },
            PolicyEntry {
                key: "a".into(),
                value: PolicyValue::Boolean(true),
            },
            PolicyEntry {
                key: "f".into(),
                value: PolicyValue::F32(-0.0),
            },
            PolicyEntry {
                key: "d".into(),
                value: PolicyValue::String("é".into()),
            },
            PolicyEntry {
                key: "b".into(),
                value: PolicyValue::Unsigned(0x0102030405060708),
            },
            PolicyEntry {
                key: "e".into(),
                value: PolicyValue::Bytes(vec![0, 255]),
            },
        ],
    };
    // Literal stream is independent of the production framing, sorting and tag writers.
    let expected = b"LATTICE_EMBED_SPACE\0\x01\0\x01\x02\0\0\0r1\x04\0\0\0bert\x04\0\0\0\
\0\x01\0\0\0a\x03\0\0\0\0\0\0\0\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\
\x01\x01\0\0\0b\x05\0\0\0\0\0\0\0\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\
\x02\x01\0\0\0v\x3f\0\0\0\0\0\0\0\0\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
\x02\x01\0\0\0w\x40\0\0\0\0\0\0\0\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff\
\x07\0\0\0\x01\0\0\0a\0\x01\x01\0\0\0b\x01\x08\x07\x06\x05\x04\x03\x02\x01\
\x01\0\0\0c\x02\xfe\xff\xff\xff\xff\xff\xff\xff\x01\0\0\0d\x03\x02\0\0\0\xc3\xa9\
\x01\0\0\0e\x04\x02\0\0\0\0\xff\x01\0\0\0f\x05\0\0\0\x80\x01\0\0\0g\x06\0\0\0\0\0\0\xf8\x3f";
    assert_eq!(descriptor.canonical_bytes().unwrap(), expected);
    let mut header = descriptor.clone();
    header.mode = IdentityMode::Header;
    let mut header_expected = expected.to_vec();
    header_expected[22] = 0;
    assert_eq!(header.canonical_bytes().unwrap(), header_expected);
    let mut revision = descriptor.clone();
    revision.output_revision = "r2".into();
    let mut revision_expected = expected.to_vec();
    revision_expected[28] = b'2';
    assert_eq!(revision.canonical_bytes().unwrap(), revision_expected);
}

#[test]
fn malformed_recipes_are_separate_from_sensitivity() {
    let mut descriptor = EmbeddingSpaceDescriptor {
        mode: IdentityMode::Full,
        output_revision: "r".into(),
        producer_family: "f".into(),
        artifacts: vec![],
        policy: vec![],
    };
    descriptor.artifacts = vec![
        Artifact {
            role: ArtifactRole::Config,
            logical_identifier: "a".into(),
            file_size: 0,
            recipe_digest: [0; 32]
        };
        2
    ];
    assert_eq!(
        descriptor.canonical_bytes(),
        Err(IdentityError::DuplicateArtifact)
    );
    descriptor.artifacts.clear();
    descriptor.policy = vec![
        PolicyEntry {
            key: "a".into(),
            value: PolicyValue::Boolean(true)
        };
        2
    ];
    assert_eq!(
        descriptor.canonical_bytes(),
        Err(IdentityError::DuplicatePolicy)
    );
    for value in [
        PolicyValue::F32(f32::NAN),
        PolicyValue::F32(f32::INFINITY),
        PolicyValue::F32(f32::NEG_INFINITY),
        PolicyValue::F64(f64::NAN),
        PolicyValue::F64(f64::INFINITY),
        PolicyValue::F64(f64::NEG_INFINITY),
    ] {
        descriptor.policy = vec![PolicyEntry {
            key: "a".into(),
            value,
        }];
        assert_eq!(
            descriptor.canonical_bytes(),
            Err(IdentityError::NonfinitePolicy)
        );
    }
    assert_eq!(length(u32::MAX as usize), Ok(u32::MAX.to_le_bytes()));
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        length(u32::MAX as usize + 1),
        Err(IdentityError::LengthOverflow)
    );
}

#[test]
fn stable_known_forms_strict_parser_and_availability() {
    let full = "lattice-embedding-space:v1:full:sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let header = "lattice-embedding-space:v1:header:sha256:0000000000000000000000000000000000000000000000000000000000000000";
    assert_eq!(
        KnownEmbeddingSpaceId::from_digest(IdentityMode::Full, [0; 32]).to_string(),
        full
    );
    assert_eq!(
        KnownEmbeddingSpaceId::from_digest(IdentityMode::Header, [0; 32]).to_string(),
        header
    );
    for value in [full, header] {
        assert_eq!(
            value.parse::<KnownEmbeddingSpaceId>().unwrap().to_string(),
            value
        );
    }
    for bad in [
        full.replace("v1", "v0"),
        full.replace("v1", "v2"),
        full.replace("v1", "v01"),
        full.replace("full", "Full"),
        full.replace("full", "unknown"),
        full.replace("sha256", "SHA256"),
        full.replace("sha256", "sha512"),
        full.replace("lattice-embedding", "Lattice-embedding"),
        format!("{full}0"),
        full[..full.len() - 1].into(),
        format!("{full}\0"),
        format!(" {full}"),
        format!("{full} "),
        format!("{full}:x"),
        full.replacen('0', "A", 1),
        "unknown".into(),
    ] {
        assert!(bad.parse::<KnownEmbeddingSpaceId>().is_err(), "{bad:?}");
    }
    let a = EmbeddingSpaceIdentity::Known(full.parse().unwrap());
    let b = EmbeddingSpaceIdentity::Known(header.parse().unwrap());
    let unavailable = EmbeddingSpaceIdentity::Unavailable;
    assert!(a.known_match(&a));
    assert!(!a.known_match(&b));
    assert!(!a.known_match(&unavailable));
    assert!(!unavailable.known_match(&unavailable));
    let changed = EmbeddingSpaceIdentity::Known(KnownEmbeddingSpaceId::from_digest(
        IdentityMode::Full,
        [1; 32],
    ));
    assert!(!a.known_match(&changed));
}

#[test]
fn stable_known_id_process_roundtrip() {
    const KEY: &str = "LATTICE_IDENTITY_ROUNDTRIP_TEST";
    const FULL: &str = "lattice-embedding-space:v1:full:sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    if let Ok(text) = std::env::var(KEY) {
        let parsed: KnownEmbeddingSpaceId = text.parse().unwrap();
        assert_eq!(parsed.to_string(), FULL);
        println!("{parsed}");
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "embedding_space::tests::stable_known_id_process_roundtrip",
            "--nocapture",
        ])
        .env(KEY, FULL)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout).unwrap().contains(FULL));
}

#[test]
fn index_artifact_uses_appended_wire_tag() {
    let descriptor = EmbeddingSpaceDescriptor {
        mode: IdentityMode::Full,
        output_revision: String::new(),
        producer_family: String::new(),
        artifacts: vec![Artifact {
            role: ArtifactRole::SafetensorsIndex,
            logical_identifier: String::new(),
            file_size: 0,
            recipe_digest: [0; 32],
        }],
        policy: vec![],
    };
    let mut expected = b"LATTICE_EMBED_SPACE\0".to_vec();
    expected.extend_from_slice(&1u16.to_le_bytes());
    expected.push(1);
    expected.extend_from_slice(&0u32.to_le_bytes());
    expected.extend_from_slice(&0u32.to_le_bytes());
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.push(3);
    expected.extend_from_slice(&0u32.to_le_bytes());
    expected.extend_from_slice(&0u64.to_le_bytes());
    expected.extend_from_slice(&[0; 32]);
    expected.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(descriptor.canonical_bytes().unwrap(), expected);
}
