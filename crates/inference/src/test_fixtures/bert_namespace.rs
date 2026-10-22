pub(super) const HIDDEN: usize = 4;
pub(super) const INTERMEDIATE: usize = 6;
pub(super) const LAYERS: usize = 2;

#[derive(Clone)]
pub(super) struct Tensor {
    pub(super) name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    pub(super) fn new(name: &str, shape: &[usize], values: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}");
        Self {
            name: name.to_owned(),
            shape: shape.to_vec(),
            values,
        }
    }
}

fn fill(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 17 + seed * 13) % 37) as f32 - 18.0) * 0.013)
        .collect()
}

pub(super) fn checkpoint(prefix: &str, pooler: bool) -> Vec<Tensor> {
    let mut tensors = vec![
        Tensor::new(
            "embeddings.word_embeddings.weight",
            &[12, HIDDEN],
            fill(12 * HIDDEN, 1),
        ),
        Tensor::new(
            "embeddings.position_embeddings.weight",
            &[32, HIDDEN],
            fill(32 * HIDDEN, 2),
        ),
        Tensor::new(
            "embeddings.token_type_embeddings.weight",
            &[2, HIDDEN],
            fill(2 * HIDDEN, 3),
        ),
        Tensor::new(
            "embeddings.LayerNorm.weight",
            &[HIDDEN],
            vec![0.91, 1.13, 0.87, 1.07],
        ),
        Tensor::new("embeddings.LayerNorm.bias", &[HIDDEN], fill(HIDDEN, 4)),
    ];
    for layer in 0..LAYERS {
        for (index, (suffix, shape)) in [
            ("attention.self.query.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.query.bias", vec![HIDDEN]),
            ("attention.self.key.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.key.bias", vec![HIDDEN]),
            ("attention.self.value.weight", vec![HIDDEN, HIDDEN]),
            ("attention.self.value.bias", vec![HIDDEN]),
            ("attention.output.dense.weight", vec![HIDDEN, HIDDEN]),
            ("attention.output.dense.bias", vec![HIDDEN]),
            ("attention.output.LayerNorm.weight", vec![HIDDEN]),
            ("attention.output.LayerNorm.bias", vec![HIDDEN]),
            ("intermediate.dense.weight", vec![INTERMEDIATE, HIDDEN]),
            ("intermediate.dense.bias", vec![INTERMEDIATE]),
            ("output.dense.weight", vec![HIDDEN, INTERMEDIATE]),
            ("output.dense.bias", vec![HIDDEN]),
            ("output.LayerNorm.weight", vec![HIDDEN]),
            ("output.LayerNorm.bias", vec![HIDDEN]),
        ]
        .into_iter()
        .enumerate()
        {
            let mut values = fill(shape.iter().product(), 5 + index + 19 * layer);
            if suffix.ends_with("LayerNorm.weight") {
                for value in &mut values {
                    *value += 1.0;
                }
            }
            tensors.push(Tensor::new(
                &format!("encoder.layer.{layer}.{suffix}"),
                &shape,
                values,
            ));
        }
    }
    if pooler {
        tensors.extend([
            Tensor::new(
                "pooler.dense.weight",
                &[HIDDEN, HIDDEN],
                vec![
                    0.31, -0.22, 0.07, 0.19, -0.11, 0.41, 0.23, -0.09, 0.17, 0.03, -0.37, 0.29,
                    0.13, -0.27, 0.11, 0.43,
                ],
            ),
            Tensor::new(
                "pooler.dense.bias",
                &[HIDDEN],
                vec![0.07, -0.13, 0.05, 0.11],
            ),
        ]);
    }
    for tensor in &mut tensors {
        tensor.name = format!("{prefix}{}", tensor.name);
    }
    tensors.extend([
        Tensor::new(
            "classifier.weight",
            &[1, HIDDEN],
            vec![0.43, -0.71, 0.29, 0.58],
        ),
        Tensor::new("classifier.bias", &[1], vec![-0.17]),
    ]);
    tensors
}

pub(super) fn safetensors_bytes(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert!(
            header
                .insert(
                    tensor.name.clone(),
                    serde_json::json!({
                        "dtype": "F32",
                        "shape": tensor.shape,
                        "data_offsets": [start, payload.len()],
                    }),
                )
                .is_none(),
            "duplicate fixture tensor {}",
            tensor.name,
        );
    }
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    header_bytes.resize(header_bytes.len().next_multiple_of(8), b' ');
    let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header_bytes);
    bytes.extend(payload);
    bytes
}

pub(super) fn rename(tensors: &mut [Tensor], old: &str, new: &str) {
    tensors
        .iter_mut()
        .find(|tensor| tensor.name == old)
        .unwrap()
        .name = new.to_owned();
}
