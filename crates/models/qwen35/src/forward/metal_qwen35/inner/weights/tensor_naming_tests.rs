use super::*;
use lattice_arch::qwen35_config::LayerType;
use lattice_metal_runtime::measurement::gpu_test_lock;
use lattice_weights::q4_weights::{Q4Block, Q4Tensor, q4_f32_to_f16, save_q4_file};
use std::path::{Path, PathBuf};

const CANONICAL: &str = "model.language_model.";
const FLAT: &str = "model.";

struct Fixture {
    dir: PathBuf,
    cfg: Qwen35Config,
    prefix: &'static str,
    tensors: Vec<Tensor>,
}

struct Tensor {
    suffix: String,
    shape: Vec<usize>,
    q4: bool,
    salt: usize,
}

impl Tensor {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn value(&self, index: usize) -> f32 {
        if self.q4 {
            (((index * 3 + index / 32 + self.salt) % 16) as f32 - 7.0) / 512.0
        } else if self.suffix.ends_with("A_log") {
            -1.0
        } else if self.suffix.ends_with("dt_bias") {
            -2.0
        } else {
            (1 + (index + self.salt) % 5) as f32 / 32.0
        }
    }

    fn quantized(&self) -> Q4Tensor {
        assert!(self.q4);
        assert_eq!(self.len() % 32, 0);
        let blocks = (0..self.len() / 32)
            .map(|block| {
                let mut packed = [0; 16];
                for (pair, byte) in packed.iter_mut().enumerate() {
                    let low = ((pair * 6 + block + self.salt) % 16) as u8;
                    let high = ((pair * 6 + 3 + block + self.salt) % 16) as u8;
                    *byte = low | (high << 4);
                }
                Q4Block {
                    scale: q4_f32_to_f16(1.0 / 512.0),
                    bias: q4_f32_to_f16(-7.0 / 512.0),
                    packed,
                }
            })
            .collect();
        Q4Tensor {
            blocks,
            shape: self.shape.clone(),
            original_len: self.len(),
        }
    }

    fn payload(&self) -> Vec<u8> {
        self.quantized()
            .blocks
            .iter()
            .flat_map(|block| {
                block
                    .scale
                    .to_le_bytes()
                    .into_iter()
                    .chain(block.bias.to_le_bytes())
                    .chain(block.packed)
            })
            .collect()
    }

    fn write(&self, path: &Path) {
        if self.q4 {
            save_q4_file(path, &self.quantized()).expect("write genuine Q4 tensor");
        } else {
            let mut bytes = b"KHF1".to_vec();
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
            for &dim in &self.shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&(self.len() as u64).to_le_bytes());
            for i in 0..self.len() {
                bytes.extend_from_slice(&q4_f32_to_f16(self.value(i)).to_le_bytes());
            }
            std::fs::write(path, bytes).expect("write genuine F16 tensor");
        }
    }
}

impl Fixture {
    fn new(prefix: &'static str, tied: bool) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = loop {
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("lattice-q4-names-{}-{id}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => break dir,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create fixture: {e}"),
            }
        };
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.hidden_size = 256;
        cfg.intermediate_size = 64;
        cfg.vocab_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 1;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 256;
        cfg.linear_num_key_heads = 1;
        cfg.linear_num_value_heads = Some(1);
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 128;
        cfg.linear_conv_kernel_dim = 4;
        cfg.full_attention_interval = 2;
        cfg.layer_types = vec![LayerType::FullAttention, LayerType::LinearAttention];
        cfg.layer_mask = vec![true, true];
        cfg.tie_word_embeddings = tied;
        cfg.max_position_embeddings = 16;
        cfg.eos_token_id = 31;
        cfg.mtp_num_hidden_layers = 0;
        cfg.quarot_rotation_seed = None;
        cfg.clone()
            .validate()
            .expect("small fixture has valid geometry");

        let mut tensors = Vec::new();
        let mut add = |suffix: String, shape: &[usize], q4: bool| {
            let salt = tensors.len() + 1;
            tensors.push(Tensor {
                suffix,
                shape: shape.to_vec(),
                q4,
                salt,
            });
        };
        add("embed_tokens.weight".into(), &[32, 256], true);
        add("norm.weight".into(), &[256], false);
        if !tied {
            add("lm_head.weight".into(), &[32, 256], true);
        }
        for layer in 0..2 {
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                add(format!("layers.{layer}.{name}"), &[256], false);
            }
            for (name, shape) in [
                ("gate_proj", [64, 256]),
                ("up_proj", [64, 256]),
                ("down_proj", [256, 64]),
            ] {
                add(format!("layers.{layer}.mlp.{name}.weight"), &shape, true);
            }
        }
        for (name, shape) in [
            ("q_proj", [512, 256]),
            ("k_proj", [256, 256]),
            ("v_proj", [256, 256]),
            ("o_proj", [256, 256]),
        ] {
            add(format!("layers.0.self_attn.{name}.weight"), &shape, true);
        }
        for name in ["q_norm", "k_norm"] {
            add(format!("layers.0.self_attn.{name}.weight"), &[256], false);
        }
        for (name, shape) in [
            ("in_proj_qkv", [384, 256]),
            ("in_proj_z", [128, 256]),
            ("in_proj_b", [1, 256]),
            ("in_proj_a", [1, 256]),
            ("out_proj", [256, 128]),
        ] {
            add(format!("layers.1.linear_attn.{name}.weight"), &shape, true);
        }
        for (name, shape) in [
            ("A_log", vec![1]),
            ("dt_bias", vec![1]),
            ("conv1d.weight", vec![384, 1, 4]),
            ("norm.weight", vec![128]),
        ] {
            add(format!("layers.1.linear_attn.{name}"), &shape, false);
        }
        let fixture = Self {
            dir,
            cfg,
            prefix,
            tensors,
        };
        for tensor in &fixture.tensors {
            tensor.write(&fixture.path(tensor, prefix));
        }
        let manifest: Vec<_> = fixture
            .tensors
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": fixture.name(t, prefix),
                    "file": fixture.path(t, prefix).file_name().unwrap().to_str().unwrap(),
                    "shape": t.shape,
                })
            })
            .collect();
        std::fs::write(
            fixture.dir.join("quantize_index.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        fixture
    }

    fn name(&self, tensor: &Tensor, prefix: &str) -> String {
        if tensor.suffix == "lm_head.weight" {
            tensor.suffix.clone()
        } else {
            format!("{prefix}{}", tensor.suffix)
        }
    }

    fn path(&self, tensor: &Tensor, prefix: &str) -> PathBuf {
        // Deliberately independent of the production path sanitizer.
        self.dir.join(format!(
            "{}.{}",
            self.name(tensor, prefix).replace('.', "_"),
            if tensor.q4 { "q4" } else { "f16" }
        ))
    }

    fn tensor(&self, suffix: &str) -> &Tensor {
        self.tensors.iter().find(|t| t.suffix == suffix).unwrap()
    }

    fn load(&self) -> Result<MetalQwen35State, String> {
        MetalQwen35State::from_q4_dir(&self.dir, &self.dir.join("tokenizer.json"), &self.cfg, 4)
    }

    fn cache_path(&self) -> PathBuf {
        let size = |suffix| {
            std::fs::metadata(self.path(self.tensor(suffix), self.prefix))
                .unwrap()
                .len()
        };
        self.dir.join(format!(
            "merged_qkvz_1_{}_{}.q4",
            size("layers.1.linear_attn.in_proj_qkv.weight"),
            size("layers.1.linear_attn.in_proj_z.weight")
        ))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Fixture {
    fn source_payloads(&self) -> Vec<Vec<u8>> {
        self.tensors
            .iter()
            .map(|tensor| std::fs::read(self.path(tensor, self.prefix)).unwrap())
            .collect()
    }

    fn assert_source_values(&self) {
        use lattice_weights::q4_weights::{
            dequantize_q4_to_f32, load_f16_tensor_file_checked, load_q4_file_checked,
        };
        for tensor in &self.tensors {
            let path = self.path(tensor, self.prefix);
            let values = if tensor.q4 {
                dequantize_q4_to_f32(&load_q4_file_checked(&path, &tensor.shape).unwrap())
            } else {
                load_f16_tensor_file_checked(&path, &tensor.shape)
                    .unwrap()
                    .0
            };
            let expected: Vec<_> = (0..tensor.len()).map(|i| tensor.value(i)).collect();
            assert_eq!(values, expected, "source {}", tensor.suffix);
        }
    }

    fn assert_cache_contents(&self) {
        let cache = self.cache_path();
        if cache.is_file() {
            let mut expected = b"KHQ4".to_vec();
            expected.extend_from_slice(&2u32.to_le_bytes());
            expected.extend_from_slice(&2u32.to_le_bytes());
            expected.extend_from_slice(&512u64.to_le_bytes());
            expected.extend_from_slice(&256u64.to_le_bytes());
            expected.extend_from_slice(&(512u64 * 256).to_le_bytes());
            expected.extend(
                self.tensor("layers.1.linear_attn.in_proj_qkv.weight")
                    .payload(),
            );
            expected.extend(
                self.tensor("layers.1.linear_attn.in_proj_z.weight")
                    .payload(),
            );
            assert_eq!(
                std::fs::read(cache).unwrap(),
                expected,
                "merged cache header and complete payload"
            );
        }
    }

    fn assert_loaded(&self, state: &MetalQwen35State) {
        assert!(matches!(state.engine.quant_format, QuantFormat::Q4_0));
        let EmbeddingTable::Q4(embedding) = &state.engine.embed_tokens else {
            panic!("native Q4 embedding required")
        };
        let mut row = vec![0.0; 256];
        embedding.write_row(1, &mut row);
        let expected: Vec<_> = (256..512)
            .map(|i| self.tensor("embed_tokens.weight").value(i))
            .collect();
        assert_eq!(row, expected);
        assert!(row.iter().any(|&v| v != 0.0));
        self.assert_q4_storage(
            &state.engine.embed_tokens_q8,
            if self.cfg.tie_word_embeddings {
                "embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        );
        self.assert_f32_storage(&state.engine.final_norm, "norm.weight");
        assert_eq!(state.engine.layer_weights.len(), 2);
        for (i, (attn, common)) in state.engine.layer_weights.iter().enumerate() {
            self.assert_f32_storage(
                &common.input_layernorm,
                &format!("layers.{i}.input_layernorm.weight"),
            );
            self.assert_f32_storage(
                &common.post_attention_layernorm,
                &format!("layers.{i}.post_attention_layernorm.weight"),
            );
            let MetalFfnWeights::Dense {
                gate_up_proj,
                down_proj,
            } = &common.ffn
            else {
                panic!("dense fixture")
            };
            let mut fused = self
                .tensor(&format!("layers.{i}.mlp.gate_proj.weight"))
                .payload();
            fused.extend(
                self.tensor(&format!("layers.{i}.mlp.up_proj.weight"))
                    .payload(),
            );
            assert_eq!(gate_up_proj.payload_offset, 0);
            assert_eq!(gate_up_proj.buffer.length(), fused.len() as u64);
            self.assert_q4_storage(down_proj, &format!("layers.{i}.mlp.down_proj.weight"));
            match attn {
                MetalLayerAttnWeights::Full(full) => {
                    assert_eq!(i, 0);
                    for (buf, name) in [
                        (&full.q_proj, "q_proj"),
                        (&full.k_proj, "k_proj"),
                        (&full.v_proj, "v_proj"),
                        (&full.o_proj, "o_proj"),
                    ] {
                        self.assert_q4_storage(buf, &format!("layers.0.self_attn.{name}.weight"));
                    }
                    self.assert_f32_storage(&full.q_norm, "layers.0.self_attn.q_norm.weight");
                    self.assert_f32_storage(&full.k_norm, "layers.0.self_attn.k_norm.weight");
                }
                MetalLayerAttnWeights::Linear(gdn) => {
                    assert_eq!(i, 1);
                    for (buf, name) in [
                        (&gdn.in_proj_qkv, "in_proj_qkv"),
                        (&gdn.in_proj_z, "in_proj_z"),
                        (&gdn.out_proj, "out_proj"),
                    ] {
                        self.assert_q4_storage(buf, &format!("layers.1.linear_attn.{name}.weight"));
                    }
                    let mut fused = self
                        .tensor("layers.1.linear_attn.in_proj_qkv.weight")
                        .payload();
                    fused.extend(
                        self.tensor("layers.1.linear_attn.in_proj_z.weight")
                            .payload(),
                    );
                    assert!(
                        gdn.in_proj_qkvz.buffer.length()
                            >= gdn.in_proj_qkvz.payload_offset + fused.len() as u64
                    );
                    for (buf, name) in
                        [(&gdn.in_proj_b, "in_proj_b"), (&gdn.in_proj_a, "in_proj_a")]
                    {
                        let tensor = self.tensor(&format!("layers.1.linear_attn.{name}.weight"));
                        assert_eq!(buf.length(), (tensor.len() * 2) as u64);
                    }
                    for (buf, name) in [
                        (&gdn.a_log, "A_log"),
                        (&gdn.dt_bias, "dt_bias"),
                        (&gdn.conv1d_weight, "conv1d.weight"),
                        (&gdn.norm_weight, "norm.weight"),
                    ] {
                        self.assert_f32_storage(buf, &format!("layers.1.linear_attn.{name}"));
                    }
                }
            }
        }
    }

    fn assert_q4_storage(&self, buf: &Q4WeightBuf, suffix: &str) {
        let expected = self.tensor(suffix).payload();
        assert!(
            buf.payload_offset > 0,
            "{suffix}: native Q4 mapping required"
        );
        assert!(
            buf.buffer.length() >= buf.payload_offset + expected.len() as u64,
            "{suffix}"
        );
    }

    fn assert_f32_storage(&self, buf: &Buffer, suffix: &str) {
        let tensor = self.tensor(suffix);
        assert_eq!(buf.length(), (tensor.len() * 4) as u64, "{suffix}");
    }
}

fn require_metal() {
    Device::system_default()
        .expect("R36 native-Q4 gate requires a Metal device; skipping is not a pass");
}

fn forward(fixture: &Fixture) -> Vec<f32> {
    fixture.assert_source_values();
    run_native_forward(fixture)
}

fn run_native_forward(fixture: &Fixture) -> Vec<f32> {
    let mut state = fixture.load().expect("load native Q4 fixture");
    fixture.assert_loaded(&state);
    fixture.assert_cache_contents();
    let MetalLayerAttnWeights::Linear(gdn) = &state.engine.layer_weights[1].0 else {
        panic!("GDN fixture layer required");
    };
    let gdn_route = if fixture.cache_path().is_dir() {
        assert_eq!(
            gdn.in_proj_qkvz.payload_offset, 0,
            "forced CPU concat buffer"
        );
        "fallback"
    } else {
        assert!(
            fixture.cache_path().is_file(),
            "fresh parallel merge must publish cache"
        );
        assert!(
            gdn.in_proj_qkvz.payload_offset > 0,
            "merged file must be mapped"
        );
        "merged"
    };
    let logits = state
        .try_forward_step(1, 0)
        .expect("native Metal forward step");
    assert_eq!(logits.len(), fixture.cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert!(logits.iter().any(|&v| v != 0.0));
    assert!(logits.windows(2).any(|v| v[0] != v[1]));
    println!(
        "R36_NATIVE_Q4_METAL prefix={} layers=2 full=1 gdn={gdn_route} logits={}",
        fixture.prefix,
        logits.len()
    );
    logits
}

fn assert_same_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "logit {i}: {a} != {e}; abs_diff={}",
            (a - e).abs()
        );
    }
}

#[test]
fn native_q4_namespaces_load_identical_weights_and_run_metal() {
    let _guard = gpu_test_lock();
    require_metal();
    let canonical = Fixture::new(CANONICAL, true);
    let flat = Fixture::new(FLAT, true);
    assert_eq!(canonical.source_payloads(), flat.source_payloads());
    let baseline = forward(&canonical);
    let repeated = forward(&canonical);
    let max_abs_diff = baseline
        .iter()
        .zip(&repeated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("R36_SAME_DIRECTORY_REPEAT max_abs_diff={max_abs_diff}");
    assert_same_logits(&repeated, &baseline);
    assert_same_logits(&forward(&flat), &baseline);

    use lattice_weights::q4_weights::{dequantize_q4_to_f32, load_q4_file_checked};
    let embedding = flat.tensor("embed_tokens.weight");
    let path = flat.path(embedding, FLAT);
    let original_payloads = flat.source_payloads();
    let mut changed_tensor = load_q4_file_checked(&path, &embedding.shape).unwrap();
    let original_values = dequantize_q4_to_f32(&changed_tensor);
    const CHANGED_ROW: usize = 2;
    const CHANGED_INDEX: usize = CHANGED_ROW * 256;
    let packed = &mut changed_tensor.blocks[CHANGED_INDEX / 32].packed[0];
    assert_ne!(*packed & 0x0f, 15);
    *packed = (*packed & 0xf0) | 15;
    save_q4_file(&path, &changed_tensor).unwrap();
    let changed_values =
        dequantize_q4_to_f32(&load_q4_file_checked(&path, &embedding.shape).unwrap());
    let changed_indices: Vec<_> = original_values
        .iter()
        .zip(&changed_values)
        .enumerate()
        .filter_map(|(i, (a, b))| (a.to_bits() != b.to_bits()).then_some(i))
        .collect();
    assert_eq!(changed_indices, [CHANGED_INDEX]);
    assert_eq!(changed_values[CHANGED_INDEX], 8.0 / 512.0);
    let changed_payloads = flat.source_payloads();
    let changed_bytes: usize = original_payloads
        .iter()
        .zip(&changed_payloads)
        .map(|(a, b)| {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b).filter(|(a, b)| a != b).count()
        })
        .sum();
    assert_eq!(changed_bytes, 1);

    // A non-input tied-embedding row changes only its output projection.
    let changed_logits = run_native_forward(&flat);
    assert_ne!(
        changed_logits[CHANGED_ROW].to_bits(),
        baseline[CHANGED_ROW].to_bits()
    );
    for (i, (changed, original)) in changed_logits.iter().zip(&baseline).enumerate() {
        if i != CHANGED_ROW {
            assert_eq!(changed.to_bits(), original.to_bits(), "unchanged logit {i}");
        }
    }
    println!("R36_SINGLE_WEIGHT_NEGATIVE row=2 column=0 code=15 changed_logits=1");
}

#[test]
fn native_q4_embedding_anchor_contract() {
    let _guard = gpu_test_lock();
    require_metal();
    let fixture = Fixture::new(FLAT, true);
    let embedding = fixture.tensor("embed_tokens.weight");
    let flat = fixture.path(embedding, FLAT);
    for tensor in &fixture.tensors {
        std::fs::copy(fixture.path(tensor, FLAT), fixture.path(tensor, CANONICAL)).unwrap();
    }
    let err = fixture.load().err().expect("both anchors must refuse");
    assert!(err.contains("checkpoint contains both"), "{err}");
    for tensor in &fixture.tensors {
        std::fs::remove_file(fixture.path(tensor, CANONICAL)).unwrap();
    }
    std::fs::remove_file(&flat).unwrap();
    let err = fixture.load().err().expect("neither anchor must refuse");
    assert!(
        err.contains("missing Q4 embedding file")
            && err.contains("model_language_model_embed_tokens_weight.q4"),
        "{err}"
    );
    std::fs::write(flat.with_extension("f16"), b"not a Q4 anchor").unwrap();
    let err = fixture
        .load()
        .err()
        .expect("F16 alone is not a supported embedding");
    assert!(err.contains("missing Q4 embedding file"), "{err}");
}

#[test]
fn native_q4_required_files_never_fall_back_to_other_namespace() {
    let _guard = gpu_test_lock();
    require_metal();
    for prefix in [FLAT, CANONICAL] {
        let fixture = Fixture::new(prefix, true);
        let opposite = if prefix == FLAT { CANONICAL } else { FLAT };
        for suffix in ["layers.0.self_attn.q_proj.weight", "norm.weight"] {
            let tensor = fixture.tensor(suffix);
            let selected = fixture.path(tensor, prefix);
            let other = fixture.path(tensor, opposite);
            std::fs::rename(&selected, &other).unwrap();
            let err = fixture
                .load()
                .err()
                .expect("required tensor in opposite namespace must refuse");
            assert!(
                err.contains(selected.file_name().unwrap().to_str().unwrap()),
                "{err}"
            );
            std::fs::rename(&other, &selected).unwrap();
        }
    }
}

#[test]
fn native_q4_unused_opposite_extras_do_not_change_results() {
    let _guard = gpu_test_lock();
    require_metal();
    for prefix in [FLAT, CANONICAL] {
        let fixture = Fixture::new(prefix, true);
        let baseline = forward(&fixture);
        let opposite = if prefix == FLAT { CANONICAL } else { FLAT };
        for suffix in ["layers.0.self_attn.q_proj.weight", "norm.weight"] {
            std::fs::write(
                fixture.path(fixture.tensor(suffix), opposite),
                b"unused malformed extra",
            )
            .unwrap();
        }
        assert_same_logits(&forward(&fixture), &baseline);
    }
}

#[test]
fn native_q4_shared_lm_head_and_tied_contract_are_unchanged() {
    let _guard = gpu_test_lock();
    require_metal();
    let flat = Fixture::new(FLAT, false);
    let canonical = Fixture::new(CANONICAL, false);
    assert_eq!(canonical.source_payloads(), flat.source_payloads());
    assert_same_logits(&forward(&flat), &forward(&canonical));
    let head = flat.path(flat.tensor("lm_head.weight"), FLAT);
    assert_eq!(head.file_name().unwrap(), "lm_head_weight.q4");
    let mut cfg = flat.cfg.clone();
    cfg.tie_word_embeddings = true;
    let err = MetalQwen35State::from_q4_dir(&flat.dir, &flat.dir.join("tokenizer.json"), &cfg, 4)
        .err()
        .expect("tied config with separate head must refuse");
    assert!(
        err.contains("tie_word_embeddings=true but lm_head.q4 is present"),
        "{err}"
    );
    std::fs::remove_file(head).unwrap();
    let err = flat
        .load()
        .err()
        .expect("untied config without shared head must refuse");
    assert!(
        err.contains("tie_word_embeddings=false but lm_head.q4 is missing"),
        "{err}"
    );
}

#[test]
fn native_q4_gdn_merge_warm_fallback_and_same_size_stale_cache() {
    let _guard = gpu_test_lock();
    require_metal();
    let mut fixture = Fixture::new(FLAT, true);
    let fresh = forward(&fixture);
    let cache = fixture.cache_path();
    assert!(cache.is_file());
    let original = std::fs::read(&cache).unwrap();
    let old_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1);
    std::fs::File::options()
        .write(true)
        .open(&cache)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old_time))
        .unwrap();
    assert_same_logits(&forward(&fixture), &fresh);
    assert_eq!(
        std::fs::metadata(&cache).unwrap().modified().unwrap(),
        old_time,
        "warm cache must be reused"
    );
    assert_eq!(std::fs::read(&cache).unwrap(), original);

    let index = fixture
        .tensors
        .iter()
        .position(|t| t.suffix == "layers.1.linear_attn.in_proj_qkv.weight")
        .unwrap();
    let path = fixture.path(&fixture.tensors[index], FLAT);
    let original_len = std::fs::metadata(&path).unwrap().len();
    fixture.tensors[index].salt += 9;
    fixture.tensors[index].write(&path);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), original_len);
    let changed = forward(&fixture);
    let rebuilt = std::fs::read(&cache).unwrap();
    assert_eq!(rebuilt.len(), original.len());
    assert_ne!(
        rebuilt, original,
        "same-size stale cache must be rebuilt from selected sources"
    );
    std::fs::remove_file(&cache).unwrap();
    std::fs::create_dir(&cache).unwrap();
    std::fs::write(
        cache.join("block-rename"),
        b"force deterministic merge publication failure",
    )
    .unwrap();
    assert_same_logits(&forward(&fixture), &changed);
    assert!(
        cache.is_dir(),
        "fallback must leave blocking directory in place"
    );
}
