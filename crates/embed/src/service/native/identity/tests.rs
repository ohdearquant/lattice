use super::super::{EmbeddingService, load_model_sync_with_bert_acquisition};
use super::*;
use lattice_weights::consumed::{DigestKind, DigestRequest};
use std::path::Path;

#[test]
fn every_bert_pooling_variant_has_an_identity_name() {
    use lattice_compute::pool::BertPooling;
    for pooling in BertPooling::ALL {
        assert!(
            pooling_name(pooling).is_some(),
            "unmapped pooling: {pooling:?}"
        );
    }
    assert_eq!(pooling_name(BertPooling::Mean), Some("mean"));
    assert_eq!(pooling_name(BertPooling::CLS), Some("cls"));
}

fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn expected_id(mode: IdentityMode, bytes: &[u8]) -> String {
    let prefix = match mode {
        IdentityMode::Full => "lattice-embedding-space:v1:full:sha256:",
        IdentityMode::Header => "lattice-embedding-space:v1:header:sha256:",
    };
    format!("{prefix}{}", hex(&hash(bytes)))
}
fn actual_id(descriptor: &EmbeddingSpaceDescriptor) -> String {
    KnownEmbeddingSpaceId::from_digest(
        descriptor.mode,
        hash(&descriptor.canonical_bytes().unwrap()),
    )
    .to_string()
}
fn known(identity: &EmbeddingSpaceIdentity) -> String {
    match identity {
        EmbeddingSpaceIdentity::Known(id) => id.to_string(),
        EmbeddingSpaceIdentity::Unavailable => panic!("expected a known successful-load recipe"),
    }
}

// Test-only oracle receives frozen input bytes and independently typed value bytes.
fn frame(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}
fn oracle(
    mode: IdentityMode,
    revision: &str,
    family: &str,
    mut artifacts: Vec<(u8, String, u64, [u8; 32])>,
    mut policy: Vec<(String, Vec<u8>)>,
) -> Vec<u8> {
    let mut out = b"LATTICE_EMBED_SPACE\0\x01\0".to_vec();
    out.push(match mode {
        IdentityMode::Header => 0,
        IdentityMode::Full => 1,
    });
    frame(&mut out, revision.as_bytes());
    frame(&mut out, family.as_bytes());
    artifacts.sort_by(|a, b| (a.0, a.1.as_bytes()).cmp(&(b.0, b.1.as_bytes())));
    out.extend_from_slice(&(artifacts.len() as u32).to_le_bytes());
    for (role, name, size, digest) in artifacts {
        out.push(role);
        frame(&mut out, name.as_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&digest);
    }
    policy.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    out.extend_from_slice(&(policy.len() as u32).to_le_bytes());
    for (name, value) in policy {
        frame(&mut out, name.as_bytes());
        out.extend_from_slice(&value);
    }
    out
}
fn string(value: &str) -> Vec<u8> {
    let mut bytes = vec![3];
    frame(&mut bytes, value.as_bytes());
    bytes
}
fn unsigned(value: u64) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.extend_from_slice(&value.to_le_bytes());
    bytes
}

#[test]
fn independent_expected_ids_cover_each_codec_component() {
    let mut descriptor = EmbeddingSpaceDescriptor {
        mode: IdentityMode::Full,
        output_revision: "r1".into(),
        producer_family: "bert".into(),
        artifacts: vec![
            Artifact {
                role: ArtifactRole::Safetensors,
                logical_identifier: "w".into(),
                file_size: 64,
                recipe_digest: [0x33; 32],
            },
            Artifact {
                role: ArtifactRole::Config,
                logical_identifier: "c".into(),
                file_size: 3,
                recipe_digest: [0x11; 32],
            },
            Artifact {
                role: ArtifactRole::Tokenizer,
                logical_identifier: "t".into(),
                file_size: 5,
                recipe_digest: [0x22; 32],
            },
        ],
        policy: vec![
            PolicyEntry {
                key: "a".into(),
                value: PolicyValue::Boolean(true),
            },
            PolicyEntry {
                key: "b".into(),
                value: PolicyValue::Unsigned(0x0102030405060708),
            },
            PolicyEntry {
                key: "c".into(),
                value: PolicyValue::Signed(-2),
            },
            PolicyEntry {
                key: "d".into(),
                value: PolicyValue::String("é".into()),
            },
            PolicyEntry {
                key: "e".into(),
                value: PolicyValue::Bytes(vec![0, 255]),
            },
            PolicyEntry {
                key: "f".into(),
                value: PolicyValue::F32(-0.0),
            },
            PolicyEntry {
                key: "g".into(),
                value: PolicyValue::F64(1.5),
            },
        ],
    };
    let records = vec![
        (0, "c".into(), 3, [0x11; 32]),
        (1, "t".into(), 5, [0x22; 32]),
        (2, "w".into(), 64, [0x33; 32]),
    ];
    let values = vec![
        ("a".into(), vec![0, 1]),
        ("b".into(), vec![1, 8, 7, 6, 5, 4, 3, 2, 1]),
        ("c".into(), vec![2, 254, 255, 255, 255, 255, 255, 255, 255]),
        ("d".into(), vec![3, 2, 0, 0, 0, 195, 169]),
        ("e".into(), vec![4, 2, 0, 0, 0, 0, 255]),
        ("f".into(), vec![5, 0, 0, 0, 128]),
        ("g".into(), vec![6, 0, 0, 0, 0, 0, 0, 248, 63]),
    ];
    let expected = oracle(
        IdentityMode::Full,
        "r1",
        "bert",
        records.clone(),
        values.clone(),
    );
    assert_eq!(
        actual_id(&descriptor),
        expected_id(IdentityMode::Full, &expected)
    );
    for index in 0..3 {
        let mut changed = descriptor.clone();
        changed.artifacts[index].file_size += 1;
        let mut expected_records = records.clone();
        let role = changed.artifacts[index].role;
        let position = match role {
            ArtifactRole::Config => 0,
            ArtifactRole::Tokenizer => 1,
            ArtifactRole::Safetensors => 2,
            ArtifactRole::SafetensorsIndex => unreachable!("fixture has no sharded index"),
        };
        expected_records[position].2 += 1;
        assert_eq!(
            actual_id(&changed),
            expected_id(
                IdentityMode::Full,
                &oracle(
                    IdentityMode::Full,
                    "r1",
                    "bert",
                    expected_records,
                    values.clone()
                )
            )
        );
        let mut changed = descriptor.clone();
        changed.artifacts[index].recipe_digest[0] = 0x55;
        let mut expected_records = records.clone();
        expected_records[position].3[0] = 0x55;
        assert_eq!(
            actual_id(&changed),
            expected_id(
                IdentityMode::Full,
                &oracle(
                    IdentityMode::Full,
                    "r1",
                    "bert",
                    expected_records,
                    values.clone()
                )
            )
        );
        let mut changed = descriptor.clone();
        changed.artifacts[index].logical_identifier.push('x');
        let mut expected_records = records.clone();
        expected_records[position].1.push('x');
        assert_eq!(
            actual_id(&changed),
            expected_id(
                IdentityMode::Full,
                &oracle(
                    IdentityMode::Full,
                    "r1",
                    "bert",
                    expected_records,
                    values.clone()
                )
            )
        );
    }
    let alternatives = [
        PolicyValue::Boolean(false),
        PolicyValue::Unsigned(0x0102030405060709),
        PolicyValue::Signed(-3),
        PolicyValue::String("ê".into()),
        PolicyValue::Bytes(vec![0, 1]),
        PolicyValue::F32(0.0),
        PolicyValue::F64(-0.0),
    ];
    let alternate_bytes = [
        vec![0, 0],
        vec![1, 9, 7, 6, 5, 4, 3, 2, 1],
        vec![2, 253, 255, 255, 255, 255, 255, 255, 255],
        vec![3, 2, 0, 0, 0, 195, 170],
        vec![4, 2, 0, 0, 0, 0, 1],
        vec![5, 0, 0, 0, 0],
        vec![6, 0, 0, 0, 0, 0, 0, 0, 128],
    ];
    for (index, value) in alternatives.into_iter().enumerate() {
        let mut changed = descriptor.clone();
        changed.policy[index].value = value;
        let mut expected_values = values.clone();
        expected_values[index].1 = alternate_bytes[index].clone();
        assert_eq!(
            actual_id(&changed),
            expected_id(
                IdentityMode::Full,
                &oracle(
                    IdentityMode::Full,
                    "r1",
                    "bert",
                    records.clone(),
                    expected_values
                )
            )
        );
    }
    descriptor.mode = IdentityMode::Header;
    assert_eq!(
        actual_id(&descriptor),
        expected_id(
            IdentityMode::Header,
            &oracle(
                IdentityMode::Header,
                "r1",
                "bert",
                records.clone(),
                values.clone()
            )
        )
    );
    descriptor.output_revision = "r2".into();
    assert_eq!(
        actual_id(&descriptor),
        expected_id(
            IdentityMode::Header,
            &oracle(
                IdentityMode::Header,
                "r2",
                "bert",
                records.clone(),
                values.clone()
            )
        )
    );
    descriptor.producer_family = "Bert".into();
    assert_eq!(
        actual_id(&descriptor),
        expected_id(
            IdentityMode::Header,
            &oracle(IdentityMode::Header, "r2", "Bert", records, values)
        )
    );
}

#[derive(Clone)]
struct Inputs {
    config: Vec<u8>,
    tokenizer: Vec<u8>,
    cap: Vec<u8>,
    weights: Vec<u8>,
    header: Vec<u8>,
}
impl Inputs {
    fn new(changed_data: bool, header_padding: usize) -> Self {
        let config = br#"{"vocab_size":6,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":6,"max_position_embeddings":8,"type_vocab_size":2,"layer_norm_eps":0.00001}"#.to_vec();
        let tokenizer=br#"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"hello":5}}}"#.to_vec();
        let cap = br#"{"model_max_length":8}"#.to_vec();
        let mut tensors: Vec<(String, Vec<usize>)> = vec![
            ("embeddings.word_embeddings.weight".into(), vec![6, 4]),
            ("embeddings.position_embeddings.weight".into(), vec![8, 4]),
            ("embeddings.token_type_embeddings.weight".into(), vec![2, 4]),
            ("embeddings.LayerNorm.weight".into(), vec![4]),
            ("embeddings.LayerNorm.bias".into(), vec![4]),
        ];
        for name in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            tensors.push((format!("encoder.layer.0.{name}.weight"), vec![4, 4]));
            tensors.push((format!("encoder.layer.0.{name}.bias"), vec![4]));
        }
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            tensors.push((format!("encoder.layer.0.{name}.weight"), vec![4]));
            tensors.push((format!("encoder.layer.0.{name}.bias"), vec![4]));
        }
        tensors.extend([
            (
                "encoder.layer.0.intermediate.dense.weight".into(),
                vec![6, 4],
            ),
            ("encoder.layer.0.intermediate.dense.bias".into(), vec![6]),
            ("encoder.layer.0.output.dense.weight".into(), vec![4, 6]),
            ("encoder.layer.0.output.dense.bias".into(), vec![4]),
        ]);
        let mut metadata = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, shape) in tensors {
            let start = payload.len();
            for index in 0..shape.iter().product::<usize>() {
                let mut value = if name.ends_with("LayerNorm.weight") {
                    1.0
                } else if name.ends_with(".bias") {
                    0.0
                } else {
                    ((index * 7 + 3) % 17) as f32 / 20.0 - 0.4
                };
                if changed_data && name == "embeddings.word_embeddings.weight" && index == 8 {
                    value += 5.0;
                }
                payload.extend_from_slice(&value.to_le_bytes());
            }
            metadata.insert(name,serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&metadata).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        header.extend(std::iter::repeat_n(b' ', (header_padding + 1) * 8));
        let mut weights = (header.len() as u64).to_le_bytes().to_vec();
        weights.extend_from_slice(&header);
        weights.extend(payload);
        Self {
            config,
            tokenizer,
            cap,
            weights,
            header,
        }
    }
    fn write(&self, dir: &Path) {
        std::fs::write(dir.join("config.json"), &self.config).unwrap();
        std::fs::write(dir.join("tokenizer.json"), &self.tokenizer).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), &self.cap).unwrap();
        std::fs::write(dir.join("model.safetensors"), &self.weights).unwrap();
    }
    fn expected(
        &self,
        mode: IdentityMode,
        format: &str,
        tokenizer_reads: &[(String, Vec<u8>)],
    ) -> String {
        self.expected_with_settings(
            mode,
            format,
            tokenizer_reads,
            expected_tokenizer_policy(format),
        )
    }
    fn expected_with_settings(
        &self,
        mode: IdentityMode,
        format: &str,
        tokenizer_reads: &[(String, Vec<u8>)],
        settings: Vec<(String, Vec<u8>)>,
    ) -> String {
        let mut recipe = (self.header.len() as u64).to_le_bytes().to_vec();
        recipe.extend_from_slice(&self.header);
        if mode == IdentityMode::Full {
            recipe.extend_from_slice(&hash(&self.weights));
        }
        let mut artifacts = vec![
            (
                0,
                "config.json".into(),
                self.config.len() as u64,
                hash(&self.config),
            ),
            (
                2,
                "model.safetensors".into(),
                self.weights.len() as u64,
                hash(&recipe),
            ),
        ];
        artifacts.extend(
            tokenizer_reads
                .iter()
                .enumerate()
                .map(|(index, (name, bytes))| {
                    (
                        1,
                        format!("read.{index}/{name}"),
                        bytes.len() as u64,
                        hash(bytes),
                    )
                }),
        );
        let mut policy = Vec::new();
        for (key, value) in [
            ("config.vocab_size", 6),
            ("config.hidden_size", 4),
            ("config.num_hidden_layers", 1),
            ("config.num_attention_heads", 2),
            ("config.intermediate_size", 6),
            ("config.max_position_embeddings", 8),
            ("config.type_vocab_size", 2),
            ("input.max_tokens", 8),
            ("output.dimension", 4),
        ] {
            policy.push((key.into(), unsigned(value)));
        }
        policy.push(("config.layer_norm_eps".into(), vec![5, 172, 197, 39, 55]));
        for (key, value) in [
            ("pooling", "cls"),
            ("normalization", "l2"),
            ("tokenizer.format", format),
            ("tokenizer.program", "lattice-tokenizer-v1"),
            ("input.truncation_side", "right"),
            ("input.program", "single-sequence-special-tokens-v1"),
            ("role.generic", ""),
            (
                "role.query",
                "Represent this sentence for searching relevant passages: ",
            ),
            ("role.passage", ""),
            (
                "role.compatibility",
                "paired-query-passage;generic-unprefixed;text-only",
            ),
            ("precision", "f32"),
            ("backend", "cpu-runtime-dispatch"),
            ("backend.target_arch", std::env::consts::ARCH),
            ("adapter", "none"),
            ("vision", "unsupported"),
        ] {
            policy.push((key.into(), string(value)));
        }
        policy.push((
            "backend.x86_gemm".into(),
            vec![0, u8::from(cfg!(feature = "x86-gemm"))],
        ));
        policy.extend(settings);
        expected_id(
            mode,
            &oracle(
                mode,
                "bert-native-output-v1",
                "bge-small-native-v1",
                artifacts,
                policy,
            ),
        )
    }
    fn expected_for_family(
        &self,
        mode: IdentityMode,
        format: &str,
        tokenizer_reads: &[(String, Vec<u8>)],
        settings: Vec<(String, Vec<u8>)>,
        family: &str,
    ) -> String {
        let mut recipe = (self.header.len() as u64).to_le_bytes().to_vec();
        recipe.extend_from_slice(&self.header);
        if mode == IdentityMode::Full {
            recipe.extend_from_slice(&hash(&self.weights));
        }
        let mut artifacts = vec![
            (
                0,
                "config.json".into(),
                self.config.len() as u64,
                hash(&self.config),
            ),
            (
                2,
                "model.safetensors".into(),
                self.weights.len() as u64,
                hash(&recipe),
            ),
        ];
        artifacts.extend(
            tokenizer_reads
                .iter()
                .enumerate()
                .map(|(index, (name, bytes))| {
                    (
                        1,
                        format!("read.{index}/{name}"),
                        bytes.len() as u64,
                        hash(bytes),
                    )
                }),
        );
        let mut policy = Vec::new();
        for (key, value) in [
            ("config.vocab_size", 6),
            ("config.hidden_size", 4),
            ("config.num_hidden_layers", 1),
            ("config.num_attention_heads", 2),
            ("config.intermediate_size", 6),
            ("config.max_position_embeddings", 8),
            ("config.type_vocab_size", 2),
            ("input.max_tokens", 8),
            ("output.dimension", 4),
        ] {
            policy.push((key.into(), unsigned(value)));
        }
        policy.push(("config.layer_norm_eps".into(), vec![5, 172, 197, 39, 55]));
        for (key, value) in [
            ("pooling", "cls"),
            ("normalization", "l2"),
            ("tokenizer.format", format),
            ("tokenizer.program", "lattice-tokenizer-v1"),
            ("input.truncation_side", "right"),
            ("input.program", "single-sequence-special-tokens-v1"),
            ("role.generic", ""),
            (
                "role.query",
                "Represent this sentence for searching relevant passages: ",
            ),
            ("role.passage", ""),
            (
                "role.compatibility",
                "paired-query-passage;generic-unprefixed;text-only",
            ),
            ("precision", "f32"),
            ("backend", "cpu-runtime-dispatch"),
            ("backend.target_arch", std::env::consts::ARCH),
            ("adapter", "none"),
            ("vision", "unsupported"),
        ] {
            policy.push((key.into(), string(value)));
        }
        policy.push((
            "backend.x86_gemm".into(),
            vec![0, u8::from(cfg!(feature = "x86-gemm"))],
        ));
        policy.extend(settings);
        expected_id(
            mode,
            &oracle(mode, "bert-native-output-v1", family, artifacts, policy),
        )
    }
    fn reads(&self) -> Vec<(String, Vec<u8>)> {
        vec![
            ("tokenizer.json".into(), self.tokenizer.clone()),
            ("tokenizer_config.json".into(), self.cap.clone()),
        ]
    }
}

async fn result(service: &NativeEmbeddingService, role: EmbeddingRole) -> IdentifiedEmbeddings {
    service
        .embed_identified(&["hello".into()], EmbeddingModel::BgeSmallEnV15, role)
        .await
        .unwrap()
}
fn assert_report(
    model: &BertModel,
    inputs: &Inputs,
    mode: IdentityMode,
    reads: &[(String, Vec<u8>)],
) {
    let report = model.load_report().unwrap();
    assert_eq!(report.config().digest(), hash(&inputs.config));
    assert_eq!(report.config().size(), inputs.config.len() as u64);
    assert_eq!(report.weights().header_bytes(), inputs.header);
    assert_eq!(report.weights().header_len(), inputs.header.len() as u64);
    assert_eq!(report.weights().file_size(), inputs.weights.len() as u64);
    assert_eq!(
        report.weights().content_digest(),
        (mode == IdentityMode::Full).then(|| hash(&inputs.weights))
    );
    assert_eq!(report.tokenizer().len(), reads.len());
    for (index, ((name, actual), (expected_name, bytes))) in
        report.tokenizer().iter().zip(reads).enumerate()
    {
        assert_eq!(name, &format!("read.{index}/{expected_name}"));
        assert_eq!(actual.digest(), hash(bytes));
        assert_eq!(actual.size(), bytes.len() as u64);
    }
}

#[tokio::test]
async fn successful_native_once_lock_binds_frozen_inputs_and_roles() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = Inputs::new(false, 0);
    inputs.write(dir.path());
    let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
    assert!(service.model.get().is_none());
    assert_eq!(service.identity_mode, IdentityMode::Full);
    let (generic, query) = tokio::join!(
        result(&service, EmbeddingRole::Generic),
        result(&service, EmbeddingRole::Query)
    );
    let passage = result(&service, EmbeddingRole::Passage).await;
    let expected = inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads());
    for output in [&generic, &query, &passage] {
        assert_eq!(known(&output.identity), expected);
        assert_eq!(output.embeddings[0].len(), 4);
    }
    let LoadedModel::Bert { model, .. } = service.model.get().unwrap().as_ref().unwrap() else {
        panic!("BERT load arm required")
    };
    assert_report(model, &inputs, IdentityMode::Full, &inputs.reads());
    let ordinary = service
        .encode_prepared(&["hello"], EmbeddingModel::BgeSmallEnV15)
        .await
        .unwrap();
    assert_eq!(generic.embeddings, ordinary);
    let replacement = Inputs::new(true, 0);
    let path = dir.path().join("replacement");
    std::fs::write(&path, &replacement.weights).unwrap();
    std::fs::rename(path, dir.path().join("model.safetensors")).unwrap();
    let retained = result(&service, EmbeddingRole::Generic).await;
    assert_eq!(known(&retained.identity), expected);
    assert_eq!(retained.embeddings, generic.embeddings);
    let fresh = NativeEmbeddingService::with_bge_small_directory(dir.path());
    let changed = result(&fresh, EmbeddingRole::Generic).await;
    assert_eq!(
        known(&changed.identity),
        replacement.expected(IdentityMode::Full, "wordpiece-json", &replacement.reads())
    );
    assert_ne!(changed.embeddings, generic.embeddings);
    assert_eq!(known(&generic.identity), expected);
}

#[tokio::test]
async fn tensor_data_only_edit_is_invisible_to_header_and_visible_to_default_full() {
    let a = Inputs::new(false, 0);
    let b = Inputs::new(true, 0);
    assert_eq!(a.header, b.header);
    assert_eq!(a.weights.len(), b.weights.len());
    assert_ne!(a.weights, b.weights);
    for mode in [IdentityMode::Full, IdentityMode::Header] {
        let da = tempfile::tempdir().unwrap();
        let db = tempfile::tempdir().unwrap();
        a.write(da.path());
        b.write(db.path());
        let sa = NativeEmbeddingService::with_bge_small_directory(da.path())
            .with_identity_mode(mode)
            .unwrap();
        let sb = NativeEmbeddingService::with_bge_small_directory(db.path())
            .with_identity_mode(mode)
            .unwrap();
        let ra = result(&sa, EmbeddingRole::Generic).await;
        let rb = result(&sb, EmbeddingRole::Generic).await;
        assert_eq!(
            known(&ra.identity),
            a.expected(mode, "wordpiece-json", &a.reads())
        );
        assert_eq!(
            known(&rb.identity),
            b.expected(mode, "wordpiece-json", &b.reads())
        );
        assert_eq!(
            ra.identity.known_match(&rb.identity),
            mode == IdentityMode::Header
        );
        assert_ne!(ra.embeddings, rb.embeddings);
    }
}

#[test]
fn digest_requests_and_same_open_replacements_bind_the_real_successful_model() {
    let original = Inputs::new(false, 0);
    for mode in [IdentityMode::Full, IdentityMode::Header] {
        for replacement in [Inputs::new(true, 0), Inputs::new(false, 2)] {
            let dir = tempfile::tempdir().unwrap();
            original.write(dir.path());
            let replacement_path = dir.path().join("next.safetensors");
            std::fs::write(&replacement_path, &replacement.weights).unwrap();
            let mut requests = Vec::new();
            let mut replaced = false;
            let model = BertModel::from_directory_with_digest_mode(
                dir.path(),
                weight_mode(mode),
                &mut |request: DigestRequest| {
                    let barrier = match mode {
                        IdentityMode::Full => DigestKind::SafetensorsContent,
                        IdentityMode::Header => DigestKind::SafetensorsRecipe,
                    };
                    if request.kind == barrier && !replaced {
                        std::fs::rename(&replacement_path, dir.path().join("model.safetensors"))
                            .unwrap();
                        replaced = true;
                    }
                    requests.push(request);
                },
            )
            .unwrap();
            assert!(replaced);
            assert_report(&model, &original, mode, &original.reads());
            let mut model = model;
            model.set_pooling(lattice_compute::pool::BertPooling::CLS);
            assert_eq!(
                known(&identify(&model, mode).unwrap()),
                original.expected(mode, "wordpiece-json", &original.reads())
            );
            let content: Vec<_> = requests
                .iter()
                .filter(|request| request.kind == DigestKind::SafetensorsContent)
                .collect();
            assert_eq!(content.len(), usize::from(mode == IdentityMode::Full));
            if let Some(request) = content.first() {
                assert_eq!(request.logical_identifier, "model.safetensors");
                assert_eq!(request.bytes, original.weights.len() as u64);
            }
            let exact: Vec<_> = requests
                .iter()
                .filter(|request| request.kind == DigestKind::ExactBytes)
                .map(|request| (request.logical_identifier.as_str(), request.bytes))
                .collect();
            assert_eq!(
                exact,
                vec![
                    ("config.json", original.config.len() as u64),
                    ("read.0/tokenizer.json", original.tokenizer.len() as u64),
                    ("read.1/tokenizer_config.json", original.cap.len() as u64)
                ]
            );
            assert_eq!(
                requests
                    .iter()
                    .filter(|request| request.kind == DigestKind::SafetensorsRecipe)
                    .count(),
                1
            );
        }
    }
}

#[test]
fn config_and_tokenizer_digest_requests_use_the_consumed_parse_buffers() {
    let inputs = Inputs::new(false, 0);
    for (request_name, file_name) in [
        ("config.json", "config.json"),
        ("read.0/tokenizer.json", "tokenizer.json"),
        ("read.1/tokenizer_config.json", "tokenizer_config.json"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        inputs.write(dir.path());
        let mut replaced = false;
        let mut model = BertModel::from_directory_with_digest_mode(
            dir.path(),
            WeightDigestMode::Full,
            &mut |request| {
                if request.kind == DigestKind::ExactBytes
                    && request.logical_identifier == request_name
                {
                    let path = dir.path().join(file_name);
                    let mut bytes = std::fs::read(&path).unwrap();
                    bytes.push(b' ');
                    let next = dir.path().join("replacement");
                    std::fs::write(&next, bytes).unwrap();
                    std::fs::rename(next, path).unwrap();
                    replaced = true;
                }
            },
        )
        .unwrap();
        assert!(replaced);
        assert_report(&model, &inputs, IdentityMode::Full, &inputs.reads());
        model.set_pooling(lattice_compute::pool::BertPooling::CLS);
        assert_eq!(
            known(&identify(&model, IdentityMode::Full).unwrap()),
            inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads())
        );
        let mut fresh = BertModel::from_directory_with_report(dir.path()).unwrap();
        assert_ne!(fresh.load_report().unwrap().config().digest(), [0; 32]);
        let fresh_bytes = std::fs::read(dir.path().join(file_name)).unwrap();
        assert_ne!(
            fresh_bytes,
            match file_name {
                "config.json" => inputs.config.clone(),
                "tokenizer.json" => inputs.tokenizer.clone(),
                _ => inputs.cap.clone(),
            }
        );
        let mut newer = inputs.clone();
        match file_name {
            "config.json" => newer.config.push(b' '),
            "tokenizer.json" => newer.tokenizer.push(b' '),
            _ => newer.cap.push(b' '),
        }
        fresh.set_pooling(lattice_compute::pool::BertPooling::CLS);
        assert_report(&fresh, &newer, IdentityMode::Full, &newer.reads());
        assert_eq!(
            known(&identify(&fresh, IdentityMode::Full).unwrap()),
            newer.expected(IdentityMode::Full, "wordpiece-json", &newer.reads())
        );
        assert_ne!(
            known(&identify(&fresh, IdentityMode::Full).unwrap()),
            inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads())
        );
    }
}

fn sentencepiece_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for (piece, kind) in [
        ("<pad>", 3),
        ("<unk>", 2),
        ("<s>", 3),
        ("</s>", 3),
        ("▁hello", 1),
    ] {
        let mut message = vec![10, piece.len() as u8];
        message.extend_from_slice(piece.as_bytes());
        message.extend_from_slice(&[21, 0, 0, 0, 0, 24, kind]);
        bytes.extend_from_slice(&[10, message.len() as u8]);
        bytes.extend(message);
    }
    bytes
}

#[tokio::test]
async fn every_selected_tokenizer_route_reports_its_exact_consumed_bytes() {
    let vocab = br#"{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4}"#.to_vec();
    let plain = b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n".to_vec();
    let wp = Inputs::new(false, 0).tokenizer;
    let bpe=br#"{"model":{"type":"BPE","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4},"merges":[]}}"#.to_vec();
    let unigram=br#"{"model":{"type":"Unigram","unk_id":1,"vocab":[["<pad>",0],["<unk>",0],["<s>",0],["</s>",0],["hello",0]]}}"#.to_vec();
    let alias = String::from_utf8(unigram.clone())
        .unwrap()
        .replace("Unigram", "SentencePieceUnigram")
        .into_bytes();
    let missing_type = br#"{"model":{}}"#.to_vec();
    type TokenizerInputFiles = Vec<(&'static str, Vec<u8>)>;
    type TokenizerCase = (&'static str, TokenizerInputFiles);
    let cases: Vec<TokenizerCase> = vec![
        ("wordpiece-json", vec![("tokenizer.json", wp)]),
        ("bpe-json", vec![("tokenizer.json", bpe)]),
        ("sentencepiece-json", vec![("tokenizer.json", unigram)]),
        ("sentencepiece-json", vec![("tokenizer.json", alias)]),
        (
            "bpe-vocab-json",
            vec![
                ("vocab.json", vocab),
                ("merges.txt", b"#version: 0.2\n".to_vec()),
            ],
        ),
        (
            "bpe-vocab-txt",
            vec![
                ("vocab.txt", plain.clone()),
                ("merges.txt", b"#version: 0.2\n".to_vec()),
            ],
        ),
        ("wordpiece-vocab-txt", vec![("vocab.txt", plain.clone())]),
        (
            "sentencepiece-model",
            vec![("tokenizer.model", sentencepiece_bytes())],
        ),
        (
            "wordpiece-vocab-txt",
            vec![("tokenizer.json", missing_type), ("vocab.txt", plain)],
        ),
    ];
    for (format, files) in cases {
        let dir = tempfile::tempdir().unwrap();
        let inputs = Inputs::new(false, 0);
        inputs.write(dir.path());
        std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();
        for (name, bytes) in &files {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        let mut reads = Vec::new();
        if let Some((_, bytes)) = files.iter().find(|(name, _)| *name == "tokenizer.json") {
            reads.push(("tokenizer.json".into(), bytes.clone()));
            reads.push(("tokenizer_config.json".into(), inputs.cap.clone()));
            if format == "wordpiece-vocab-txt" {
                reads.push(("tokenizer_config.json".into(), inputs.cap.clone()));
            }
        } else {
            reads.push(("tokenizer_config.json".into(), inputs.cap.clone()));
        }
        reads.extend(
            files
                .iter()
                .filter(|(name, _)| *name != "tokenizer.json")
                .map(|(name, bytes)| (name.to_string(), bytes.clone())),
        );
        let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
        let output = result(&service, EmbeddingRole::Generic).await;
        assert_eq!(
            known(&output.identity),
            inputs.expected(IdentityMode::Full, format, &reads),
            "{format}"
        );
        let LoadedModel::Bert { model, .. } = service.model.get().unwrap().as_ref().unwrap() else {
            panic!("BERT required")
        };
        assert_eq!(model.load_report().unwrap().tokenizer_format(), format);
        assert_report(model, &inputs, IdentityMode::Full, &reads);
    }
}

#[tokio::test]
async fn lazy_failed_loads_and_missing_origins_do_not_mint_known_recipes() {
    let dir = tempfile::tempdir().unwrap();
    let a = Inputs::new(false, 0);
    a.write(dir.path());
    let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
    let b = Inputs::new(true, 0);
    b.write(dir.path());
    let output = result(&service, EmbeddingRole::Generic).await;
    assert_eq!(
        known(&output.identity),
        b.expected(IdentityMode::Full, "wordpiece-json", &b.reads())
    );
    let missing = tempfile::tempdir().unwrap();
    let failed = NativeEmbeddingService::with_bge_small_directory(missing.path());
    assert!(
        failed
            .embed_identified(
                &["hello".into()],
                EmbeddingModel::BgeSmallEnV15,
                EmbeddingRole::Generic
            )
            .await
            .is_err()
    );
    a.write(missing.path());
    assert!(
        failed
            .embed_identified(
                &["hello".into()],
                EmbeddingModel::BgeSmallEnV15,
                EmbeddingRole::Generic
            )
            .await
            .is_err()
    );
    let repaired = NativeEmbeddingService::with_bge_small_directory(missing.path());
    assert_eq!(
        known(&result(&repaired, EmbeddingRole::Generic).await.identity),
        a.expected(IdentityMode::Full, "wordpiece-json", &a.reads())
    );
    let memory = BertModel::from_bytes(
        a.weights.clone(),
        std::str::from_utf8(&a.config).unwrap(),
        std::str::from_utf8(&a.tokenizer).unwrap(),
    )
    .unwrap();
    assert!(memory.load_report().is_none());
    assert!(matches!(
        identify(&memory, IdentityMode::Full).unwrap(),
        EmbeddingSpaceIdentity::Unavailable
    ));
}

#[tokio::test]
async fn valid_header_size_config_and_tokenizer_changes_have_independent_ids() {
    let original = Inputs::new(false, 0);
    for which in 0..4 {
        let mut changed = Inputs::new(false, if which == 0 { 1 } else { 0 });
        if which == 1 {
            changed.config.push(b' ');
        }
        if which == 2 {
            changed.tokenizer.push(b' ');
        }
        if which == 3 {
            changed.cap.extend_from_slice(b" ");
        }
        let dir = tempfile::tempdir().unwrap();
        changed.write(dir.path());
        let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
        let output = result(&service, EmbeddingRole::Generic).await;
        assert_eq!(
            known(&output.identity),
            changed.expected(IdentityMode::Full, "wordpiece-json", &changed.reads())
        );
        assert_ne!(
            known(&output.identity),
            original.expected(IdentityMode::Full, "wordpiece-json", &original.reads())
        );
        if which == 0 {
            assert_ne!(changed.header.len(), original.header.len());
            assert_ne!(changed.weights.len(), original.weights.len());
        }
    }
}

fn expected_tokenizer_policy(format: &str) -> Vec<(String, Vec<u8>)> {
    let mut settings = vec![("max_seq_len".into(), unsigned(8))];
    let id = |key: &str, value: Option<u64>| {
        let mut entries = vec![(format!("{key}.present"), vec![0, u8::from(value.is_some())])];
        if let Some(value) = value {
            entries.push((format!("{key}.value"), unsigned(value)));
        }
        entries
    };
    if format.starts_with("wordpiece") {
        settings.push(("vocab_size".into(), unsigned(6)));
        settings.push(("program".into(), string("wordpiece-single-v1")));
        for (key, value) in [("cls", 2), ("sep", 3), ("pad", 0), ("unk", 1), ("mask", 4)] {
            settings.extend(id(key, Some(value)));
        }
        settings.push(("added.count".into(), unsigned(0)));
    } else if format.starts_with("bpe") {
        settings.push((
            "vocab_size".into(),
            unsigned(if format == "bpe-vocab-txt" { 6 } else { 5 }),
        ));
        settings.push(("program".into(), string("bpe-single-v1")));
        for key in ["add_bos", "add_eos"] {
            settings.push((key.into(), vec![0, 0]));
        }
        for (key, value) in [
            ("pad", Some(0)),
            ("unk", Some(1)),
            ("bos", None),
            ("eos", None),
        ] {
            settings.extend(id(key, value));
        }
        settings.push(("pretokenizer".into(), string("gpt4-regex")));
        settings.push(("special.count".into(), unsigned(0)));
    } else {
        settings.push(("vocab_size".into(), unsigned(5)));
        settings.push(("program".into(), string("sentencepiece-single-v1")));
        settings.push(("min_score".into(), vec![5, 0, 0, 0, 0]));
        for (key, value) in [
            ("add_bos", false),
            ("add_eos", false),
            ("dummy_prefix", true),
            ("remove_extra_whitespaces", true),
            ("escape_whitespaces", true),
        ] {
            settings.push((key.into(), vec![0, u8::from(value)]));
        }
        let native = format == "sentencepiece-model";
        for (key, value) in [
            ("pad", Some(if native { 1 } else { 0 })),
            ("unk", Some(1)),
            ("bos", if native { None } else { Some(2) }),
            ("eos", if native { None } else { Some(3) }),
        ] {
            settings.extend(id(key, value));
        }
        settings.push(("byte_fallback.count".into(), unsigned(0)));
        settings.push(("added.count".into(), unsigned(0)));
    }
    settings
        .into_iter()
        .map(|(key, value)| (format!("tokenizer.effective.{key}"), value))
        .collect()
}

#[test]
fn literal_canonical_golden_has_independent_expected_id() {
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
    assert_eq!(
        actual_id(&descriptor),
        expected_id(IdentityMode::Full, expected)
    );
    let mut header = descriptor.clone();
    header.mode = IdentityMode::Header;
    let mut header_expected = expected.to_vec();
    header_expected[22] = 0;
    assert_eq!(header.canonical_bytes().unwrap(), header_expected);
    assert_eq!(
        actual_id(&header),
        expected_id(IdentityMode::Header, &header_expected)
    );
    let mut revision = descriptor.clone();
    revision.output_revision = "r2".into();
    let mut revision_expected = expected.to_vec();
    revision_expected[28] = b'2';
    assert_eq!(revision.canonical_bytes().unwrap(), revision_expected);
    assert_eq!(
        actual_id(&revision),
        expected_id(IdentityMode::Full, &revision_expected)
    );
}

#[tokio::test]
async fn raw_header_reformatting_and_ignored_cap_reads_are_consumed_inputs() {
    let original = Inputs::new(false, 0);
    let mut header_changed = original.clone();
    assert_eq!(header_changed.header.first(), Some(&b'{'));
    assert_eq!(header_changed.header.pop(), Some(b' '));
    header_changed.header.insert(1, b' ');
    let header_end = 8 + header_changed.header.len();
    header_changed.weights[8..header_end].copy_from_slice(&header_changed.header);
    assert_ne!(header_changed.header, original.header);
    assert_eq!(header_changed.weights.len(), original.weights.len());
    assert_eq!(header_changed.header.len(), original.header.len());
    for mode in [IdentityMode::Full, IdentityMode::Header] {
        let dir = tempfile::tempdir().unwrap();
        header_changed.write(dir.path());
        let service = NativeEmbeddingService::with_bge_small_directory(dir.path())
            .with_identity_mode(mode)
            .unwrap();
        let output = result(&service, EmbeddingRole::Generic).await;
        assert_eq!(
            known(&output.identity),
            header_changed.expected(mode, "wordpiece-json", &header_changed.reads())
        );
        assert_ne!(
            known(&output.identity),
            original.expected(mode, "wordpiece-json", &original.reads())
        );
    }
    let mut outputs = Vec::new();
    let mut ids = Vec::new();
    for ignored in [
        b"{".as_slice(),
        b"[".as_slice(),
        b"\xff".as_slice(),
        b"\xfe".as_slice(),
    ] {
        let mut inputs = original.clone();
        inputs.cap = ignored.to_vec();
        let dir = tempfile::tempdir().unwrap();
        inputs.write(dir.path());
        let reads = vec![
            ("tokenizer.json".into(), inputs.tokenizer.clone()),
            ("tokenizer_config.json".into(), inputs.cap.clone()),
            ("config.json".into(), inputs.config.clone()),
        ];
        let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
        let output = result(&service, EmbeddingRole::Generic).await;
        assert_eq!(
            known(&output.identity),
            inputs.expected(IdentityMode::Full, "wordpiece-json", &reads)
        );
        let LoadedModel::Bert { model, .. } = service.model.get().unwrap().as_ref().unwrap() else {
            panic!("BERT required")
        };
        assert_report(model, &inputs, IdentityMode::Full, &reads);
        ids.push(known(&output.identity));
        outputs.push(output.embeddings);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[2], outputs[3]);
    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[2], ids[3]);
}

#[tokio::test]
async fn nonempty_tokenizer_maps_bind_each_indexed_effective_setting() {
    let cases = [
        ("wordpiece-json", br#"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"hello":5}},"added_tokens":[{"id":5,"content":"hello"}]}"#.as_slice(), "added", 5),
        ("bpe-json", br#"{"model":{"type":"BPE","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4},"merges":[]},"added_tokens":[{"id":4,"content":"hello","special":true}]}"#.as_slice(), "special", 4),
        ("sentencepiece-json", br#"{"model":{"type":"Unigram","unk_id":1,"byte_fallback":true,"vocab":[["<pad>",0],["<unk>",0],["<s>",0],["</s>",0],["<0x41>",0]]},"added_tokens":[{"id":4,"content":"hello"}]}"#.as_slice(), "added", 4),
    ];
    for (format, tokenizer, map, token_id) in cases {
        let mut inputs = Inputs::new(false, 0);
        inputs.tokenizer = tokenizer.to_vec();
        let mut settings = expected_tokenizer_policy(format);
        let count_key = format!("tokenizer.effective.{map}.count");
        let count = settings
            .iter_mut()
            .find(|(key, _)| key == &count_key)
            .unwrap();
        count.1 = unsigned(1);
        settings.extend([
            (format!("tokenizer.effective.{map}.0.text"), string("hello")),
            (
                format!("tokenizer.effective.{map}.0.id"),
                unsigned(token_id),
            ),
        ]);
        if format == "sentencepiece-json" {
            let count = settings
                .iter_mut()
                .find(|(key, _)| key == "tokenizer.effective.byte_fallback.count")
                .unwrap();
            count.1 = unsigned(1);
            settings.extend([
                (
                    "tokenizer.effective.byte_fallback.0.byte".into(),
                    unsigned(65),
                ),
                ("tokenizer.effective.byte_fallback.0.id".into(), unsigned(4)),
            ]);
        }
        let expected =
            inputs.expected_with_settings(IdentityMode::Full, format, &inputs.reads(), settings);
        let dir = tempfile::tempdir().unwrap();
        inputs.write(dir.path());
        let service = NativeEmbeddingService::with_bge_small_directory(dir.path());
        let output = service
            .embed_identified(
                &["hello".into(), "A".into()],
                EmbeddingModel::BgeSmallEnV15,
                EmbeddingRole::Generic,
            )
            .await
            .unwrap();
        assert_eq!(known(&output.identity), expected, "{format}");
        assert_eq!(output.embeddings.len(), 2);
        assert!(output.embeddings.iter().all(|vector| vector.len() == 4));
        let LoadedModel::Bert { model, .. } = service.model.get().unwrap().as_ref().unwrap() else {
            panic!("BERT required")
        };
        assert_report(model, &inputs, IdentityMode::Full, &inputs.reads());
        use lattice_tokenizer::common::TokenizerPolicyValue as V;
        assert!(
            model
                .load_report()
                .unwrap()
                .tokenizer_settings()
                .contains(&(format!("{map}.0.text"), V::String("hello".into())))
        );
        assert!(
            model
                .load_report()
                .unwrap()
                .tokenizer_settings()
                .contains(&(format!("{map}.0.id"), V::Unsigned(token_id)))
        );
        if format == "sentencepiece-json" {
            assert!(
                model
                    .load_report()
                    .unwrap()
                    .tokenizer_settings()
                    .contains(&("byte_fallback.0.byte".into(), V::Unsigned(65)))
            );
            assert!(
                model
                    .load_report()
                    .unwrap()
                    .tokenizer_settings()
                    .contains(&("byte_fallback.0.id".into(), V::Unsigned(4)))
            );
        }
    }
}

#[test]
fn canceled_queued_initialization_freezes_identity_mode_at_start() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let blocker = runtime.spawn_blocking(move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    started_rx.recv().unwrap();

    let directory = tempfile::tempdir().unwrap();
    let inputs = Inputs::new(false, 0);
    inputs.write(directory.path());
    let service = NativeEmbeddingService::with_bge_small_directory(directory.path());
    let polled_pending = runtime.block_on(async {
        let mut load = Box::pin(service.ensure_loaded());
        std::future::poll_fn(|context| {
            std::task::Poll::Ready(std::future::Future::poll(load.as_mut(), context).is_pending())
        })
        .await
    });
    let unpublished = service.model.get().is_none();
    // The consuming builder drops its input on error; this handle observes the
    // original cell without manufacturing another model or report.
    let observer = NativeEmbeddingService {
        model: service.model.clone(),
        model_config: service.model_config,
        identity_mode: service.identity_mode,
        initialization_started: std::sync::atomic::AtomicBool::new(true),
        model_directory: service.model_directory.clone(),
    };
    let change = service.with_identity_mode(IdentityMode::Header);
    let rejected = change.is_err();

    release_tx.send(()).unwrap();
    let (blocker_result, original_output) = runtime.block_on(async {
        let blocker_result = blocker.await;
        let output = observer
            .embed_identified(
                &["hello".into()],
                EmbeddingModel::BgeSmallEnV15,
                EmbeddingRole::Generic,
            )
            .await;
        (blocker_result, output)
    });
    let changed_output = change.ok().map(|changed| {
        runtime.block_on(async {
            changed
                .embed_identified(
                    &["hello".into()],
                    EmbeddingModel::BgeSmallEnV15,
                    EmbeddingRole::Generic,
                )
                .await
        })
    });

    assert!(blocker_result.is_ok());
    assert!(polled_pending);
    assert!(unpublished);
    assert!(
        rejected,
        "mode selection must freeze before queued initialization"
    );
    assert!(changed_output.is_none());
    let original_output = original_output.unwrap();
    assert_eq!(
        known(&original_output.identity),
        inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads())
    );
    assert_eq!(original_output.embeddings[0].len(), 4);

    let before_start = NativeEmbeddingService::with_bge_small_directory(directory.path())
        .with_identity_mode(IdentityMode::Header)
        .unwrap();
    let header = runtime.block_on(result(&before_start, EmbeddingRole::Generic));
    assert_eq!(
        known(&header.identity),
        inputs.expected(IdentityMode::Header, "wordpiece-json", &inputs.reads())
    );
    assert!(before_start.with_identity_mode(IdentityMode::Full).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_identity_is_computed_once_per_loaded_service_and_used_by_concurrent_calls() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let directory = tempfile::tempdir().unwrap();
    Inputs::new(false, 0).write(directory.path());
    let builds = std::sync::Arc::new(AtomicUsize::new(0));
    let sentinel = EmbeddingSpaceIdentity::Known(KnownEmbeddingSpaceId::from_digest(
        IdentityMode::Full,
        [0x5a; 32],
    ));
    let service = std::sync::Arc::new(NativeEmbeddingService::with_bge_small_directory(
        directory.path(),
    ));
    let loaded = load_model_sync_with_bert_acquisition(
        service.model_config,
        service.identity_mode,
        None,
        |_| {
            BertModel::from_directory_with_report(directory.path())
                .map_err(|error| error.to_string())
        },
        |model, mode| {
            builds.fetch_add(1, Ordering::SeqCst);
            assert_eq!(mode, IdentityMode::Full);
            assert_eq!(model.pooling(), lattice_compute::pool::BertPooling::CLS);
            Ok(sentinel.clone())
        },
    )
    .unwrap();
    assert!(service.model.set(Ok(loaded)).is_ok());
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(8));
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let service = service.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            service
                .embed_identified(
                    &["hello".into()],
                    EmbeddingModel::BgeSmallEnV15,
                    EmbeddingRole::Generic,
                )
                .await
                .unwrap()
                .identity
        }));
    }
    for task in tasks {
        let identity = task.await.unwrap();
        assert!(identity.known_match(&sentinel));
    }
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    let fresh = NativeEmbeddingService::with_bge_small_directory(directory.path());
    let fresh_output = result(&fresh, EmbeddingRole::Generic).await;
    assert!(!fresh_output.identity.known_match(&sentinel));
}

#[tokio::test]
async fn identity_error_is_retained_without_failing_ordinary_embedding() {
    let directory = tempfile::tempdir().unwrap();
    Inputs::new(false, 0).write(directory.path());
    let service = NativeEmbeddingService::with_bge_small_directory(directory.path());
    let loaded = load_model_sync_with_bert_acquisition(
        service.model_config,
        service.identity_mode,
        None,
        |_| {
            BertModel::from_directory_with_report(directory.path())
                .map_err(|error| error.to_string())
        },
        |_, _| Err("identity test failure".into()),
    )
    .unwrap();
    assert!(service.model.set(Ok(loaded)).is_ok());
    service.ensure_loaded().await.unwrap();
    service
        .embed(&["hello".into()], EmbeddingModel::BgeSmallEnV15)
        .await
        .unwrap();
    for _ in 0..2 {
        let error = service
            .embed_identified(
                &["hello".into()],
                EmbeddingModel::BgeSmallEnV15,
                EmbeddingRole::Generic,
            )
            .await
            .err()
            .unwrap();
        assert!(
            matches!(error, EmbedError::InferenceFailed(ref message) if message == "identity test failure")
        );
    }
}

#[tokio::test]
async fn concurrent_initialization_publishes_one_identity() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let directory = tempfile::tempdir().unwrap();
    let inputs = Inputs::new(false, 0);
    inputs.write(directory.path());
    let service = NativeEmbeddingService::with_bge_small_directory(directory.path());
    let barrier = std::sync::Barrier::new(8);
    let builds = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                barrier.wait();
                let loaded = service.model.get_or_init(|| {
                    load_model_sync_with_bert_acquisition(
                        service.model_config,
                        service.identity_mode,
                        None,
                        |_| {
                            BertModel::from_directory_with_report(directory.path())
                                .map_err(|error| error.to_string())
                        },
                        |model, mode| {
                            builds.fetch_add(1, Ordering::SeqCst);
                            identify(model, mode)
                        },
                    )
                });
                assert!(loaded.is_ok());
            });
        }
    });
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    let output = result(&service, EmbeddingRole::Generic).await;
    assert_eq!(
        known(&output.identity),
        inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads())
    );
}

#[tokio::test]
async fn service_without_a_load_report_retains_unavailable_identity() {
    let directory = tempfile::tempdir().unwrap();
    Inputs::new(false, 0).write(directory.path());
    let service = NativeEmbeddingService::with_bge_small_directory(directory.path());
    let loaded = load_model_sync_with_bert_acquisition(
        service.model_config,
        service.identity_mode,
        None,
        |_| BertModel::from_directory(directory.path()).map_err(|error| error.to_string()),
        identify,
    )
    .unwrap();
    assert!(service.model.set(Ok(loaded)).is_ok());
    let output = result(&service, EmbeddingRole::Generic).await;
    assert!(matches!(
        output.identity,
        EmbeddingSpaceIdentity::Unavailable
    ));
}

fn bge_variant_service(
    dir: &Path,
    family: EmbeddingModel,
    mode: IdentityMode,
) -> NativeEmbeddingService {
    let mut service = NativeEmbeddingService::with_model(family)
        .with_identity_mode(mode)
        .unwrap();
    service.model_directory = Some(dir.to_path_buf());
    service
}

async fn variant_result(
    service: &NativeEmbeddingService,
    family: EmbeddingModel,
    role: EmbeddingRole,
) -> IdentifiedEmbeddings {
    service
        .embed_identified(&["hello".into()], family, role)
        .await
        .unwrap()
}

#[tokio::test]
async fn bge_base_large_report_full_identity_without_changing_small_golden() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = Inputs::new(false, 0);
    inputs.write(dir.path());
    let small = NativeEmbeddingService::with_bge_small_directory(dir.path());
    let small_result = result(&small, EmbeddingRole::Generic).await;
    let small_golden = inputs.expected(IdentityMode::Full, "wordpiece-json", &inputs.reads());
    assert_eq!(known(&small_result.identity), small_golden);
    let mut ids = Vec::new();
    for (family, domain) in [
        (EmbeddingModel::BgeBaseEnV15, "bge-base-native-v1"),
        (EmbeddingModel::BgeLargeEnV15, "bge-large-native-v1"),
    ] {
        let service = bge_variant_service(dir.path(), family, IdentityMode::Full);
        let (generic, concurrent) = tokio::join!(
            variant_result(&service, family, EmbeddingRole::Generic),
            variant_result(&service, family, EmbeddingRole::Generic),
        );
        assert_eq!(generic.embeddings, concurrent.embeddings);
        assert!(generic.identity.known_match(&concurrent.identity));
        let query = variant_result(&service, family, EmbeddingRole::Query).await;
        let passage = variant_result(&service, family, EmbeddingRole::Passage).await;
        let expected = inputs.expected_for_family(
            IdentityMode::Full,
            "wordpiece-json",
            &inputs.reads(),
            expected_tokenizer_policy("wordpiece-json"),
            domain,
        );
        assert_eq!(known(&generic.identity), expected);
        assert_eq!(known(&query.identity), expected);
        assert_eq!(known(&passage.identity), expected);
        assert_ne!(expected, small_golden);
        assert_eq!(generic.embeddings[0].len(), 4);
        assert!(generic.embeddings[0].iter().all(|value| value.is_finite()));
        let mut reference = BertModel::from_directory(dir.path()).unwrap();
        reference.set_pooling(lattice_compute::pool::BertPooling::CLS);
        assert_eq!(generic.embeddings[0], reference.encode("hello").unwrap());
        let query_text = format!("{}hello", family.query_instruction().unwrap());
        assert_eq!(query.embeddings[0], reference.encode(&query_text).unwrap());
        assert_eq!(passage.embeddings[0], reference.encode("hello").unwrap());
        let LoadedModel::Bert { model, .. } = service.model.get().unwrap().as_ref().unwrap() else {
            panic!("expected BERT");
        };
        assert_eq!(model.pooling(), lattice_compute::pool::BertPooling::CLS);
        ids.push(expected);
        let header = variant_result(
            &bge_variant_service(dir.path(), family, IdentityMode::Header),
            family,
            EmbeddingRole::Generic,
        )
        .await;
        assert!(matches!(
            header.identity,
            EmbeddingSpaceIdentity::Unavailable
        ));
    }
    assert_ne!(ids[0], ids[1]);
    assert_eq!(
        known(&result(&small, EmbeddingRole::Generic).await.identity),
        small_golden
    );
}

#[tokio::test]
async fn bge_variant_full_identity_tracks_each_consumed_input() {
    let baseline = Inputs::new(false, 0);
    for family in [EmbeddingModel::BgeBaseEnV15, EmbeddingModel::BgeLargeEnV15] {
        let dir = tempfile::tempdir().unwrap();
        baseline.write(dir.path());
        let first = variant_result(
            &bge_variant_service(dir.path(), family, IdentityMode::Full),
            family,
            EmbeddingRole::Generic,
        )
        .await;
        for input in ["config", "tokenizer", "weights"] {
            let dir = tempfile::tempdir().unwrap();
            let mut changed = Inputs::new(input == "weights", 0);
            if input == "config" {
                changed.config.push(b' ');
            }
            if input == "tokenizer" {
                changed.tokenizer.push(b' ');
            }
            changed.write(dir.path());
            let other = variant_result(
                &bge_variant_service(dir.path(), family, IdentityMode::Full),
                family,
                EmbeddingRole::Generic,
            )
            .await;
            assert!(
                !first.identity.known_match(&other.identity),
                "{family:?} {input}"
            );
            assert_eq!(other.embeddings[0].len(), 4);
        }
    }
}

#[tokio::test]
async fn bge_variant_missing_config_returns_unavailable() {
    use lattice_inference::bert_pooling_fixture::write_checkpoint;
    let parent = tempfile::tempdir().unwrap();
    let dir = parent.path().join("checkpoint");
    write_checkpoint(&dir).unwrap();
    std::fs::remove_file(dir.join("config.json")).unwrap();
    for family in [EmbeddingModel::BgeBaseEnV15, EmbeddingModel::BgeLargeEnV15] {
        let model = bge_variant_service(&dir, family, IdentityMode::Full);
        let output = variant_result(&model, family, EmbeddingRole::Generic).await;
        assert!(matches!(
            output.identity,
            EmbeddingSpaceIdentity::Unavailable
        ));
        assert_eq!(output.embeddings[0].len(), 384);
    }
}

#[tokio::test]
async fn bge_variant_failed_load_returns_no_identified_vector() {
    let dir = tempfile::tempdir().unwrap();
    Inputs::new(false, 0).write(dir.path());
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    for family in [EmbeddingModel::BgeBaseEnV15, EmbeddingModel::BgeLargeEnV15] {
        let service = bge_variant_service(dir.path(), family, IdentityMode::Full);
        let result = service
            .embed_identified(&["hello".into()], family, EmbeddingRole::Generic)
            .await;
        assert!(result.is_err());
    }
}
