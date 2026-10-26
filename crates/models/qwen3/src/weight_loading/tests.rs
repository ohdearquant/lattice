use super::*;
use serde_json::json;
use std::path::Path;

#[derive(Clone)]
struct Entry {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
    dtype: &'static str,
}

fn entry(name: &str, shape: &[usize], values: &[f32]) -> Entry {
    Entry {
        name: name.into(),
        shape: shape.into(),
        values: values.into(),
        dtype: "F32",
    }
}

fn tensors() -> Vec<Entry> {
    let mut entries = vec![
        entry("embed_tokens.weight", &[3, 2], &[1., 2., 3., 4., 5., 6.]),
        entry("norm.weight", &[2], &[7., 8.]),
    ];
    for (layer, q, k, v, o, qn, kn, iln, gate, up, down, post) in [
        (
            0,
            [11., 12., 13., 14., 15., 16.],
            [21., 22.],
            [31., 32.],
            [41., 42., 43., 44., 45., 46.],
            51.,
            52.,
            [61., 62.],
            [71., 72., 73., 74., 75., 76.],
            [81., 82., 83., 84., 85., 86.],
            [91., 92., 93., 94., 95., 96.],
            [101., 102.],
        ),
        (
            1,
            [111., 112., 113., 114., 115., 116.],
            [121., 122.],
            [131., 132.],
            [141., 142., 143., 144., 145., 146.],
            151.,
            152.,
            [161., 162.],
            [171., 172., 173., 174., 175., 176.],
            [181., 182., 183., 184., 185., 186.],
            [191., 192., 193., 194., 195., 196.],
            [201., 202.],
        ),
    ] {
        let p = format!("layers.{layer}");
        for (suffix, shape, values) in [
            ("self_attn.q_proj.weight", vec![3, 2], q.to_vec()),
            ("self_attn.k_proj.weight", vec![1, 2], k.to_vec()),
            ("self_attn.v_proj.weight", vec![1, 2], v.to_vec()),
            ("self_attn.o_proj.weight", vec![2, 3], o.to_vec()),
            ("self_attn.q_norm.weight", vec![1], vec![qn]),
            ("self_attn.k_norm.weight", vec![1], vec![kn]),
            ("input_layernorm.weight", vec![2], iln.to_vec()),
            ("mlp.gate_proj.weight", vec![3, 2], gate.to_vec()),
            ("mlp.up_proj.weight", vec![3, 2], up.to_vec()),
            ("mlp.down_proj.weight", vec![2, 3], down.to_vec()),
            ("post_attention_layernorm.weight", vec![2], post.to_vec()),
        ] {
            entries.push(entry(&format!("{p}.{suffix}"), &shape, &values));
        }
    }
    entries
}

fn write_file(path: &Path, entries: &[Entry]) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for tensor in entries {
        let start = payload.len();
        for value in &tensor.values {
            if tensor.dtype == "BF16" {
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            } else {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        header.insert(
            tensor.name.clone(),
            json!({"dtype": tensor.dtype,
            "shape": tensor.shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    std::fs::write(path, file).unwrap();
}

struct Checkpoint {
    dir: tempfile::TempDir,
}
impl Checkpoint {
    fn new(entries: &[Entry]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        write_file(&dir.path().join("model.safetensors"), entries);
        let mut map = serde_json::Map::new();
        for shard in 0..2 {
            let selected: Vec<_> = entries
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == shard)
                .map(|(_, entry)| entry.clone())
                .collect();
            let name = format!("shard-{shard}.safetensors");
            write_file(&dir.path().join(&name), &selected);
            for tensor in selected {
                map.insert(tensor.name, json!(name));
            }
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"metadata": {"total_size": 0}, "weight_map": map})).unwrap(),
        )
        .unwrap();
        Self { dir }
    }
    fn single(&self) -> SafetensorsFile {
        SafetensorsFile::open(&self.dir.path().join("model.safetensors")).unwrap()
    }
    fn sharded(&self) -> ShardedSafetensors {
        ShardedSafetensors::open_index(&self.dir.path().join("model.safetensors.index.json"))
            .unwrap()
    }
}
fn borrowed(source: &SafetensorsFile) -> Result<QwenWeights<'_>, InferenceError> {
    load_weights(source, 2, 2, 3, 1, 1, 3)
}
fn owned(source: &mut ShardedSafetensors) -> Result<OwnedQwenWeights, InferenceError> {
    load_owned_weights(source, 2, 2, 3, 1, 1, 3)
}
fn assert_literal_weights(weights: &QwenWeights<'_>) {
    assert_eq!(
        (weights.embed_tokens.rows, weights.embed_tokens.cols),
        (3, 2)
    );
    assert_eq!(weights.embed_tokens.data, &[1., 2., 3., 4., 5., 6.]);
    assert_eq!(weights.norm_weight.len, 2);
    assert_eq!(weights.norm_weight.data, &[7., 8.]);
    assert_eq!(weights.layers.len(), 2);
    let expected = tensors();
    for (i, layer) in weights.layers.iter().enumerate() {
        for (suffix, data, shape) in [
            (
                "self_attn.q_proj.weight",
                layer.q_proj_weight.data,
                vec![layer.q_proj_weight.rows, layer.q_proj_weight.cols],
            ),
            (
                "self_attn.k_proj.weight",
                layer.k_proj_weight.data,
                vec![layer.k_proj_weight.rows, layer.k_proj_weight.cols],
            ),
            (
                "self_attn.v_proj.weight",
                layer.v_proj_weight.data,
                vec![layer.v_proj_weight.rows, layer.v_proj_weight.cols],
            ),
            (
                "self_attn.o_proj.weight",
                layer.o_proj_weight.data,
                vec![layer.o_proj_weight.rows, layer.o_proj_weight.cols],
            ),
            (
                "self_attn.q_norm.weight",
                layer.q_norm_weight.data,
                vec![layer.q_norm_weight.len],
            ),
            (
                "self_attn.k_norm.weight",
                layer.k_norm_weight.data,
                vec![layer.k_norm_weight.len],
            ),
            (
                "input_layernorm.weight",
                layer.input_layernorm_weight.data,
                vec![layer.input_layernorm_weight.len],
            ),
            (
                "mlp.gate_proj.weight",
                layer.gate_proj_weight.data,
                vec![layer.gate_proj_weight.rows, layer.gate_proj_weight.cols],
            ),
            (
                "mlp.up_proj.weight",
                layer.up_proj_weight.data,
                vec![layer.up_proj_weight.rows, layer.up_proj_weight.cols],
            ),
            (
                "mlp.down_proj.weight",
                layer.down_proj_weight.data,
                vec![layer.down_proj_weight.rows, layer.down_proj_weight.cols],
            ),
            (
                "post_attention_layernorm.weight",
                layer.post_attention_layernorm_weight.data,
                vec![layer.post_attention_layernorm_weight.len],
            ),
        ] {
            let name = format!("layers.{i}.{suffix}");
            let tensor = expected.iter().find(|tensor| tensor.name == name).unwrap();
            assert_eq!(data, tensor.values, "{name}");
            assert_eq!(shape, tensor.shape, "{name}");
        }
        assert_eq!(layer.qkv_out_dim, 5);
        assert_eq!(layer.gate_up_out_dim, 6);
    }
    assert_eq!(
        weights.layers[0].fused_qkv,
        [11., 12., 13., 14., 15., 16., 21., 22., 31., 32.]
    );
    assert_eq!(
        weights.layers[1].fused_qkv,
        [111., 112., 113., 114., 115., 116., 121., 122., 131., 132.]
    );
    assert_eq!(
        weights.layers[0].fused_gate_up,
        [71., 72., 73., 74., 75., 76., 81., 82., 83., 84., 85., 86.]
    );
    assert_eq!(
        weights.layers[1].fused_gate_up,
        [
            171., 172., 173., 174., 175., 176., 181., 182., 183., 184., 185., 186.
        ]
    );
}
fn assert_shape(error: InferenceError, name: &str, expected: &[usize], actual: &[usize]) {
    match error {
        InferenceError::ShapeMismatch {
            name: got,
            expected: want,
            actual: got_shape,
        } => {
            assert_eq!(got, name);
            assert_eq!(want, expected);
            assert_eq!(got_shape, actual);
        }
        other => panic!("expected shape error for {name}, got {other:?}"),
    }
}

#[test]
fn borrowed_literal_shapes_pointers_and_fusion_order() {
    let checkpoint = Checkpoint::new(&tensors());
    let source = checkpoint.single();
    let weights = borrowed(&source).unwrap();
    assert_literal_weights(&weights);
    assert_eq!(
        weights.embed_tokens.data.as_ptr(),
        source
            .get_f32_tensor("embed_tokens.weight")
            .unwrap()
            .0
            .as_ptr()
    );
    for (i, layer) in weights.layers.iter().enumerate() {
        assert_eq!(
            layer.q_proj_weight.data.as_ptr(),
            source
                .get_f32_tensor(&format!("layers.{i}.self_attn.q_proj.weight"))
                .unwrap()
                .0
                .as_ptr()
        );
    }
}
#[test]
fn owned_views_survive_source_drop_and_rebuild_fused_buffers() {
    let checkpoint = Checkpoint::new(&tensors());
    let mut source = checkpoint.sharded();
    let owner = Box::new(owned(&mut source).unwrap());
    drop(source);
    drop(checkpoint);
    let first = owner.assemble_borrowed_weights();
    let second = owner.assemble_borrowed_weights();
    assert_literal_weights(&first);
    assert_literal_weights(&second);
    assert_eq!(
        first.embed_tokens.data.as_ptr(),
        owner.backing.embed_tokens.as_ptr()
    );
    assert_eq!(
        first.norm_weight.data.as_ptr(),
        owner.backing.norm_weight.as_ptr()
    );
    for i in 0..2 {
        assert_eq!(
            first.layers[i].q_proj_weight.data.as_ptr(),
            owner.backing.q_proj[i].as_ptr()
        );
        assert_eq!(
            first.layers[i].down_proj_weight.data.as_ptr(),
            owner.backing.down_proj[i].as_ptr()
        );
        assert_ne!(
            first.layers[i].fused_qkv.as_ptr(),
            second.layers[i].fused_qkv.as_ptr()
        );
        assert_ne!(
            first.layers[i].fused_gate_up.as_ptr(),
            second.layers[i].fused_gate_up.as_ptr()
        );
    }
}
#[test]
#[allow(deprecated)]
fn legacy_loaders_match_independent_literal_contract() {
    let checkpoint = Checkpoint::new(&tensors());
    let source = checkpoint.single();
    assert_literal_weights(&source.load_qwen_weights(2, 2, 3, 1, 1, 3).unwrap());
    assert_literal_weights(&borrowed(&source).unwrap());
    let mut source = checkpoint.sharded();
    let (backing, legacy) = source.load_qwen_weights_owned(2, 2, 3, 1, 1, 3).unwrap();
    assert_literal_weights(&legacy);
    drop(legacy);
    drop(backing);
    let owner = owned(&mut source).unwrap();
    assert_literal_weights(&owner.assemble_borrowed_weights());
}
#[test]
fn missing_names_do_not_fall_back_to_a_prefixed_namespace() {
    let mut entries = tensors();
    entries[0].name = "model.embed_tokens.weight".into();
    let checkpoint = Checkpoint::new(&entries);
    assert!(
        matches!(borrowed(&checkpoint.single()), Err(InferenceError::MissingTensor(name)) if name == "embed_tokens.weight")
    );
    assert!(
        matches!(owned(&mut checkpoint.sharded()), Err(InferenceError::MissingTensor(name)) if name == "embed_tokens.weight")
    );
}
#[test]
fn every_raw_tensor_shape_is_checked_before_assembly() {
    for (name, expected, actual) in [
        ("embed_tokens.weight", vec![6, 2], vec![6]),
        ("norm.weight", vec![2], vec![1, 2]),
        ("layers.0.self_attn.q_proj.weight", vec![3, 2], vec![2, 3]),
        ("layers.0.self_attn.k_proj.weight", vec![1, 2], vec![2, 1]),
        ("layers.0.self_attn.v_proj.weight", vec![1, 2], vec![2]),
        ("layers.0.self_attn.o_proj.weight", vec![2, 3], vec![3, 2]),
        ("layers.0.self_attn.q_norm.weight", vec![1], vec![1, 1]),
        ("layers.0.self_attn.k_norm.weight", vec![1], vec![1, 1]),
        ("layers.0.input_layernorm.weight", vec![2], vec![1, 2]),
        ("layers.0.mlp.gate_proj.weight", vec![3, 2], vec![2, 3]),
        ("layers.0.mlp.up_proj.weight", vec![3, 2], vec![6]),
        ("layers.0.mlp.down_proj.weight", vec![2, 3], vec![3, 2]),
        (
            "layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1, 2],
        ),
    ] {
        let mut entries = tensors();
        entries
            .iter_mut()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape = actual.clone();
        let checkpoint = Checkpoint::new(&entries);
        assert_shape(
            borrowed(&checkpoint.single()).unwrap_err(),
            name,
            &expected,
            &actual,
        );
        assert_shape(
            owned(&mut checkpoint.sharded()).unwrap_err(),
            name,
            &expected,
            &actual,
        );
    }
}
#[test]
fn dtype_refusal_precedes_rank_check_for_materialized_reads() {
    let mut entries = tensors();
    let norm = entries
        .iter_mut()
        .find(|tensor| tensor.name == "norm.weight")
        .unwrap();
    norm.dtype = "I32";
    norm.shape = vec![1, 2];
    let checkpoint = Checkpoint::new(&entries);
    for error in [
        borrowed(&checkpoint.single()).unwrap_err(),
        owned(&mut checkpoint.sharded()).unwrap_err(),
    ] {
        assert!(
            matches!(error, InferenceError::InvalidSafetensors(message) if message.contains("tensor norm.weight has unsupported dtype I32"))
        );
    }
}
#[test]
fn borrowed_and_owned_recipes_preserve_distinct_first_refusals() {
    let mut entries = tensors();
    entries.retain(|tensor| {
        tensor.name != "layers.0.mlp.gate_proj.weight"
            && tensor.name != "layers.0.self_attn.o_proj.weight"
    });
    let checkpoint = Checkpoint::new(&entries);
    assert!(
        matches!(borrowed(&checkpoint.single()), Err(InferenceError::MissingTensor(name)) if name == "layers.0.mlp.gate_proj.weight")
    );
    assert!(
        matches!(owned(&mut checkpoint.sharded()), Err(InferenceError::MissingTensor(name)) if name == "layers.0.self_attn.o_proj.weight")
    );
}
#[test]
fn shape_refusal_precedes_next_read_in_both_recipes() {
    let mut entries = tensors();
    entries
        .iter_mut()
        .find(|tensor| tensor.name == "layers.0.self_attn.q_proj.weight")
        .unwrap()
        .shape = vec![2, 3];
    entries.retain(|tensor| tensor.name != "layers.0.self_attn.k_proj.weight");
    let checkpoint = Checkpoint::new(&entries);
    assert_shape(
        borrowed(&checkpoint.single()).unwrap_err(),
        "layers.0.self_attn.q_proj.weight",
        &[3, 2],
        &[2, 3],
    );
    assert_shape(
        owned(&mut checkpoint.sharded()).unwrap_err(),
        "layers.0.self_attn.q_proj.weight",
        &[3, 2],
        &[2, 3],
    );
}
#[test]
fn globals_precede_missing_layers_and_zero_layers_are_empty() {
    let checkpoint = Checkpoint::new(&[entry(
        "embed_tokens.weight",
        &[3, 2],
        &[1., 2., 3., 4., 5., 6.],
    )]);
    assert!(
        matches!(borrowed(&checkpoint.single()), Err(InferenceError::MissingTensor(name)) if name == "norm.weight")
    );
    assert!(
        matches!(owned(&mut checkpoint.sharded()), Err(InferenceError::MissingTensor(name)) if name == "norm.weight")
    );
    let checkpoint = Checkpoint::new(&tensors()[..2]);
    let file = checkpoint.single();
    let weights = load_weights(&file, 0, 2, 3, 1, 1, 3).unwrap();
    assert!(weights.layers.is_empty());
    assert_eq!(weights.embed_tokens.data, &[1., 2., 3., 4., 5., 6.]);
    let owner = load_owned_weights(&mut checkpoint.sharded(), 0, 2, 3, 1, 1, 3).unwrap();
    let weights = owner.assemble_borrowed_weights();
    assert!(weights.layers.is_empty());
    assert_eq!(weights.norm_weight.data, &[7., 8.]);
}
#[test]
fn converted_owned_getter_transfers_the_cached_allocation() {
    let mut tensor = entry("value", &[2], &[1., -2.]);
    tensor.dtype = "BF16";
    let checkpoint = Checkpoint::new(&[tensor]);
    let mut source = checkpoint.single();
    let pointer = source.get_f32_tensor("value").unwrap().0.as_ptr();
    let (values, shape) = TensorSource::get_f32_tensor_owned(&mut source, "value").unwrap();
    assert_eq!(shape, [2]);
    assert_eq!(values, [1., -2.]);
    assert_eq!(values.as_ptr(), pointer);
}
#[test]
#[cfg(target_endian = "little")]
fn f32_owned_getter_copies_the_aligned_borrowed_view() {
    let checkpoint = Checkpoint::new(&[entry("value", &[2], &[1., -2.])]);
    let mut source = checkpoint.single();
    let pointer = source.get_f32_tensor("value").unwrap().0.as_ptr();
    assert_eq!(pointer.align_offset(std::mem::align_of::<f32>()), 0);
    let (values, shape) = TensorSource::get_f32_tensor_owned(&mut source, "value").unwrap();
    assert_eq!(shape, [2]);
    assert_eq!(values, [1., -2.]);
    assert_ne!(values.as_ptr(), pointer);
    assert_eq!(source.get_f32_tensor("value").unwrap().0, &[1., -2.]);
}
#[test]
fn public_loader_and_owner_paths_keep_the_receiver_borrow() {
    use crate::weight_loading::{
        OwnedQwenWeights as PublicOwner, load_owned_weights as public_owned,
        load_weights as public_borrowed,
    };
    fn borrow(source: &lattice_weights::SafetensorsFile) -> QwenWeights<'_> {
        public_borrowed(source, 0, 2, 3, 1, 1, 3).unwrap()
    }
    fn assemble(owner: &PublicOwner) -> QwenWeights<'_> {
        owner.assemble_borrowed_weights()
    }
    let checkpoint = Checkpoint::new(&tensors()[..2]);
    assert_eq!(borrow(&checkpoint.single()).norm_weight.data, &[7., 8.]);
    let owner = public_owned(&mut checkpoint.sharded(), 0, 2, 3, 1, 1, 3).unwrap();
    assert_eq!(
        assemble(&owner).embed_tokens.data,
        &[1., 2., 3., 4., 5., 6.]
    );
}

#[test]
fn embedding_metadata_precedes_borrowed_materialization_only() {
    let mut entries = tensors();
    entries[0].shape = vec![6];
    entries[0].dtype = "I32";
    let checkpoint = Checkpoint::new(&entries);
    assert_shape(
        borrowed(&checkpoint.single()).unwrap_err(),
        "embed_tokens.weight",
        &[6, 2],
        &[6],
    );
    assert!(
        matches!(owned(&mut checkpoint.sharded()), Err(InferenceError::InvalidSafetensors(message)) if message.contains("tensor embed_tokens.weight has unsupported dtype I32"))
    );
}

#[test]
fn every_adjacent_read_keeps_its_recipe_precedence() {
    let borrowed_order = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
    ];
    let owned_order = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
    ];
    for (is_owned, order) in [(false, borrowed_order), (true, owned_order)] {
        for pair in order.windows(2) {
            let first = format!("layers.0.{}", pair[0]);
            let second = format!("layers.0.{}", pair[1]);
            let entries: Vec<_> = tensors()
                .into_iter()
                .filter(|tensor| tensor.name != first && tensor.name != second)
                .collect();
            let checkpoint = Checkpoint::new(&entries);
            let error = if is_owned {
                owned(&mut checkpoint.sharded()).unwrap_err()
            } else {
                borrowed(&checkpoint.single()).unwrap_err()
            };
            assert!(matches!(error, InferenceError::MissingTensor(name) if name == first));
        }
    }
}
