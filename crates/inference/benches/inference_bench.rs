use std::sync::OnceLock;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use lattice_inference::attention::standard::{AttentionBuffers, multi_head_attention};
use lattice_inference::forward::cpu::{
    add_bias, add_bias_gelu, gelu, layer_norm, matmul_bt, softmax_attention,
};
use lattice_inference::pool::{l2_normalize, mean_pool};
use lattice_inference::weights::{BertWeights, Tensor1D, Tensor2D, TransformerLayerWeights};
use lattice_inference::{BertConfig, WordPieceTokenizer};

const HIDDEN_SIZE: usize = 384;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 32;
const INTERMEDIATE_SIZE: usize = 1536;
const NUM_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 30_522;
const MAX_SEQ_LEN: usize = 512;
const TYPE_VOCAB_SIZE: usize = 2;
const LAYER_NORM_EPS: f32 = 1e-12;

#[derive(Debug, Clone)]
struct SyntheticInput {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Vec<u32>,
    seq_len: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
struct SyntheticLayerStorage {
    tensors: Vec<&'static [f32]>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct SyntheticModelStorage {
    tensors: Vec<&'static [f32]>,
    layers: Vec<SyntheticLayerStorage>,
}

#[derive(Debug)]
struct ModelFixture {
    config: BertConfig,
    weights: BertWeights<'static>,
    _storage: SyntheticModelStorage,
}

#[derive(Debug)]
struct TokenizerFixture {
    tokenizer: WordPieceTokenizer,
    short: String,
    medium: String,
    long: String,
    batch_medium: Vec<String>,
    short_tokens: usize,
    medium_tokens: usize,
    long_tokens: usize,
    batch_medium_tokens: usize,
}

fn bge_small_config() -> BertConfig {
    BertConfig {
        vocab_size: VOCAB_SIZE,
        hidden_size: HIDDEN_SIZE,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: INTERMEDIATE_SIZE,
        max_position_embeddings: MAX_SEQ_LEN,
        type_vocab_size: TYPE_VOCAB_SIZE,
        layer_norm_eps: LAYER_NORM_EPS,
    }
}

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn random_vec(len: usize) -> Vec<f32> {
    random_vec_with_seed(len, 0x1234_5678)
}

fn random_vec_with_seed(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xA341_316C;
    }

    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let bits = xorshift32(&mut state);
        let unit = bits as f32 / u32::MAX as f32;
        out.push(unit * 0.04 - 0.02);
    }
    out
}

fn random_bias_with_seed(len: usize, seed: u32) -> Vec<f32> {
    random_vec_with_seed(len, seed)
        .into_iter()
        .map(|v| v * 0.5)
        .collect()
}

fn random_layer_norm_scale(len: usize, seed: u32) -> Vec<f32> {
    random_vec_with_seed(len, seed)
        .into_iter()
        .map(|v| 1.0 + v * 0.5)
        .collect()
}

fn leak_slice(data: Vec<f32>) -> &'static [f32] {
    Box::leak(data.into_boxed_slice())
}

fn record_layer_tensor(storage: &mut SyntheticLayerStorage, data: Vec<f32>) -> &'static [f32] {
    let leaked = leak_slice(data);
    storage.tensors.push(leaked);
    leaked
}

fn record_model_tensor(storage: &mut SyntheticModelStorage, data: Vec<f32>) -> &'static [f32] {
    let leaked = leak_slice(data);
    storage.tensors.push(leaked);
    leaked
}

fn tensor1d(data: &'static [f32]) -> Tensor1D<'static> {
    Tensor1D {
        data,
        len: data.len(),
    }
}

fn tensor2d(data: &'static [f32], rows: usize, cols: usize) -> Tensor2D<'static> {
    Tensor2D { data, rows, cols }
}

fn synthetic_layer_weights(
    hidden_size: usize,
    intermediate_size: usize,
) -> (TransformerLayerWeights<'static>, SyntheticLayerStorage) {
    synthetic_layer_weights_with_seed(hidden_size, intermediate_size, 0xC0DE_0001)
}

fn synthetic_layer_weights_with_seed(
    hidden_size: usize,
    intermediate_size: usize,
    seed: u32,
) -> (TransformerLayerWeights<'static>, SyntheticLayerStorage) {
    let mut storage = SyntheticLayerStorage {
        tensors: Vec::new(),
    };

    let query_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(hidden_size * hidden_size, seed ^ 0x0001),
    );
    let query_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x0002),
    );
    let key_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(hidden_size * hidden_size, seed ^ 0x0003),
    );
    let key_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x0004),
    );
    let value_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(hidden_size * hidden_size, seed ^ 0x0005),
    );
    let value_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x0006),
    );
    let attn_output_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(hidden_size * hidden_size, seed ^ 0x0007),
    );
    let attn_output_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x0008),
    );
    let attn_layer_norm_weight = record_layer_tensor(
        &mut storage,
        random_layer_norm_scale(hidden_size, seed ^ 0x0009),
    );
    let attn_layer_norm_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x000A),
    );
    let ffn_intermediate_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(intermediate_size * hidden_size, seed ^ 0x000B),
    );
    let ffn_intermediate_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(intermediate_size, seed ^ 0x000C),
    );
    let ffn_output_weight = record_layer_tensor(
        &mut storage,
        random_vec_with_seed(hidden_size * intermediate_size, seed ^ 0x000D),
    );
    let ffn_output_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x000E),
    );
    let ffn_layer_norm_weight = record_layer_tensor(
        &mut storage,
        random_layer_norm_scale(hidden_size, seed ^ 0x000F),
    );
    let ffn_layer_norm_bias = record_layer_tensor(
        &mut storage,
        random_bias_with_seed(hidden_size, seed ^ 0x0010),
    );

    let layer = TransformerLayerWeights {
        query_weight: tensor2d(query_weight, hidden_size, hidden_size),
        query_bias: tensor1d(query_bias),
        key_weight: tensor2d(key_weight, hidden_size, hidden_size),
        key_bias: tensor1d(key_bias),
        value_weight: tensor2d(value_weight, hidden_size, hidden_size),
        value_bias: tensor1d(value_bias),
        attn_output_weight: tensor2d(attn_output_weight, hidden_size, hidden_size),
        attn_output_bias: tensor1d(attn_output_bias),
        attn_layer_norm_weight: tensor1d(attn_layer_norm_weight),
        attn_layer_norm_bias: tensor1d(attn_layer_norm_bias),
        ffn_intermediate_weight: tensor2d(ffn_intermediate_weight, intermediate_size, hidden_size),
        ffn_intermediate_bias: tensor1d(ffn_intermediate_bias),
        ffn_output_weight: tensor2d(ffn_output_weight, hidden_size, intermediate_size),
        ffn_output_bias: tensor1d(ffn_output_bias),
        ffn_layer_norm_weight: tensor1d(ffn_layer_norm_weight),
        ffn_layer_norm_bias: tensor1d(ffn_layer_norm_bias),
    };

    (layer, storage)
}

fn synthetic_bert_weights(config: &BertConfig) -> (BertWeights<'static>, SyntheticModelStorage) {
    let mut storage = SyntheticModelStorage {
        tensors: Vec::new(),
        layers: Vec::with_capacity(config.num_hidden_layers),
    };

    let word_embeddings = record_model_tensor(
        &mut storage,
        random_vec_with_seed(config.vocab_size * config.hidden_size, 0xBEE1_0001),
    );
    let position_embeddings = record_model_tensor(
        &mut storage,
        random_vec_with_seed(
            config.max_position_embeddings * config.hidden_size,
            0xBEE1_0002,
        ),
    );
    let token_type_embeddings = record_model_tensor(
        &mut storage,
        random_vec_with_seed(config.type_vocab_size * config.hidden_size, 0xBEE1_0003),
    );
    let embedding_layer_norm_weight = record_model_tensor(
        &mut storage,
        random_layer_norm_scale(config.hidden_size, 0xBEE1_0004),
    );
    let embedding_layer_norm_bias = record_model_tensor(
        &mut storage,
        random_bias_with_seed(config.hidden_size, 0xBEE1_0005),
    );
    let pooler_weight = record_model_tensor(
        &mut storage,
        random_vec_with_seed(config.hidden_size * config.hidden_size, 0xBEE1_0006),
    );
    let pooler_bias = record_model_tensor(
        &mut storage,
        random_bias_with_seed(config.hidden_size, 0xBEE1_0007),
    );

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let seed = 0xCAFE_0000u32 ^ ((layer_idx as u32 + 1).wrapping_mul(0x1F1F_0101));
        let (layer, layer_storage) =
            synthetic_layer_weights_with_seed(config.hidden_size, config.intermediate_size, seed);
        layers.push(layer);
        storage.layers.push(layer_storage);
    }

    let weights = BertWeights {
        word_embeddings: tensor2d(word_embeddings, config.vocab_size, config.hidden_size),
        position_embeddings: tensor2d(
            position_embeddings,
            config.max_position_embeddings,
            config.hidden_size,
        ),
        token_type_embeddings: tensor2d(
            token_type_embeddings,
            config.type_vocab_size,
            config.hidden_size,
        ),
        embedding_layer_norm_weight: tensor1d(embedding_layer_norm_weight),
        embedding_layer_norm_bias: tensor1d(embedding_layer_norm_bias),
        layers,
        pooler_weight: tensor2d(pooler_weight, config.hidden_size, config.hidden_size),
        pooler_bias: tensor1d(pooler_bias),
    };

    (weights, storage)
}

fn synthetic_vocab_text(vocab_size: usize) -> String {
    assert!(
        vocab_size > 103,
        "vocab_size must leave room for BERT special tokens"
    );

    let mut vocab = vec![String::new(); vocab_size];
    vocab[0] = "[PAD]".to_string();
    vocab[100] = "[UNK]".to_string();
    vocab[101] = "[CLS]".to_string();
    vocab[102] = "[SEP]".to_string();
    vocab[103] = "[MASK]".to_string();

    let mut tok_idx = 0usize;
    let mut sub_idx = 0usize;
    for slot in vocab.iter_mut() {
        if slot.is_empty() {
            if tok_idx < 20_000 {
                *slot = format!("tok{tok_idx}");
                tok_idx += 1;
            } else {
                *slot = format!("##sub{sub_idx}");
                sub_idx += 1;
            }
        }
    }

    vocab.join("\n")
}

fn synthetic_sentence(word_count: usize, offset: usize) -> String {
    let mut sentence = String::with_capacity(word_count * 8);
    for idx in 0..word_count {
        if idx > 0 {
            sentence.push(' ');
        }
        let token_idx = (offset + idx) % 20_000;
        sentence.push_str(&format!("tok{token_idx}"));
    }
    sentence
}

fn synthetic_input(config: &BertConfig, seq_len: usize, token_offset: usize) -> SyntheticInput {
    assert!(seq_len >= 2, "seq_len must be at least 2 for [CLS]/[SEP]");
    assert!(seq_len <= config.max_position_embeddings);

    let mut input_ids = vec![0u32; seq_len];
    input_ids[0] = 101;
    for (i, token) in input_ids[1..seq_len - 1].iter_mut().enumerate() {
        *token = 104 + ((token_offset + i) % (config.vocab_size - 104)) as u32;
    }
    input_ids[seq_len - 1] = 102;

    SyntheticInput {
        input_ids,
        attention_mask: vec![1u32; seq_len],
        token_type_ids: vec![0u32; seq_len],
        seq_len,
    }
}

fn synthetic_batch_inputs(
    config: &BertConfig,
    batch_size: usize,
    seq_len: usize,
) -> Vec<SyntheticInput> {
    (0..batch_size)
        .map(|batch_idx| synthetic_input(config, seq_len, batch_idx * seq_len))
        .collect()
}

fn forward_pass(
    config: &BertConfig,
    weights: &BertWeights<'_>,
    input_ids: &[u32],
    attention_mask: &[u32],
    token_type_ids: &[u32],
    seq_len: usize,
    buffers: &mut AttentionBuffers,
) -> Vec<f32> {
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let used_hidden = seq_len * hidden_size;

    debug_assert_eq!(input_ids.len(), seq_len);
    debug_assert_eq!(attention_mask.len(), seq_len);
    debug_assert_eq!(token_type_ids.len(), seq_len);
    assert!(
        seq_len <= config.max_position_embeddings,
        "sequence length {seq_len} exceeds max_position_embeddings {}",
        config.max_position_embeddings
    );

    let mut hidden = vec![0.0f32; used_hidden];

    for i in 0..seq_len {
        let tok_id = input_ids[i] as usize;
        let typ_id = token_type_ids[i] as usize;
        let pos_id = i;

        debug_assert!(tok_id < weights.word_embeddings.rows);
        debug_assert!(typ_id < weights.token_type_embeddings.rows);
        debug_assert!(pos_id < weights.position_embeddings.rows);

        let tok_row =
            &weights.word_embeddings.data[tok_id * hidden_size..(tok_id + 1) * hidden_size];
        let pos_row =
            &weights.position_embeddings.data[pos_id * hidden_size..(pos_id + 1) * hidden_size];
        let typ_row =
            &weights.token_type_embeddings.data[typ_id * hidden_size..(typ_id + 1) * hidden_size];
        let out_row = &mut hidden[i * hidden_size..(i + 1) * hidden_size];

        for d in 0..hidden_size {
            out_row[d] = tok_row[d] + pos_row[d] + typ_row[d];
        }
    }

    layer_norm(
        &mut hidden,
        weights.embedding_layer_norm_weight.data,
        weights.embedding_layer_norm_bias.data,
        hidden_size,
        config.layer_norm_eps,
    );

    for layer in &weights.layers {
        let mut attn_output = multi_head_attention(
            &hidden,
            layer,
            attention_mask,
            seq_len,
            hidden_size,
            config.num_attention_heads,
            hidden_size / config.num_attention_heads,
            buffers,
        );

        for i in 0..used_hidden {
            attn_output[i] += hidden[i];
        }
        layer_norm(
            &mut attn_output,
            layer.attn_layer_norm_weight.data,
            layer.attn_layer_norm_bias.data,
            hidden_size,
            config.layer_norm_eps,
        );
        hidden = attn_output;

        let used_intermediate = seq_len * intermediate_size;
        {
            let ffn_intermediate = &mut buffers.ffn_intermediate[..used_intermediate];
            matmul_bt(
                &hidden,
                layer.ffn_intermediate_weight.data,
                ffn_intermediate,
                seq_len,
                hidden_size,
                intermediate_size,
            );
            add_bias_gelu(
                ffn_intermediate,
                layer.ffn_intermediate_bias.data,
                intermediate_size,
            );
        }

        {
            let ffn_intermediate = &buffers.ffn_intermediate[..used_intermediate];
            let temp = &mut buffers.temp[..used_hidden];
            matmul_bt(
                ffn_intermediate,
                layer.ffn_output_weight.data,
                temp,
                seq_len,
                intermediate_size,
                hidden_size,
            );
            add_bias(temp, layer.ffn_output_bias.data, hidden_size);
            for i in 0..used_hidden {
                temp[i] += hidden[i];
            }
            layer_norm(
                temp,
                layer.ffn_layer_norm_weight.data,
                layer.ffn_layer_norm_bias.data,
                hidden_size,
                config.layer_norm_eps,
            );
            hidden.copy_from_slice(temp);
        }
    }

    hidden
}

fn encode_with_weights(
    config: &BertConfig,
    weights: &BertWeights<'_>,
    input: &SyntheticInput,
    buffers: &mut AttentionBuffers,
) -> Vec<f32> {
    let hidden_states = forward_pass(
        config,
        weights,
        &input.input_ids,
        &input.attention_mask,
        &input.token_type_ids,
        input.seq_len,
        buffers,
    );

    let mut pooled = mean_pool(
        &hidden_states,
        &input.attention_mask,
        input.seq_len,
        config.hidden_size,
    );
    l2_normalize(&mut pooled);
    pooled
}

fn encode_batch_with_weights(
    config: &BertConfig,
    weights: &BertWeights<'_>,
    inputs: &[SyntheticInput],
    buffers: &mut AttentionBuffers,
) -> Vec<Vec<f32>> {
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        outputs.push(encode_with_weights(config, weights, input, buffers));
    }
    outputs
}

fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    2u64 * m as u64 * k as u64 * n as u64
}

fn model_fixture() -> &'static ModelFixture {
    static MODEL: OnceLock<ModelFixture> = OnceLock::new();
    MODEL.get_or_init(|| {
        let config = bge_small_config();
        let (weights, storage) = synthetic_bert_weights(&config);
        ModelFixture {
            config,
            weights,
            _storage: storage,
        }
    })
}

fn tokenizer_fixture() -> &'static TokenizerFixture {
    static TOKENIZER: OnceLock<TokenizerFixture> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let vocab_text = synthetic_vocab_text(VOCAB_SIZE);
        let tokenizer = WordPieceTokenizer::from_str(&vocab_text)
            .expect("synthetic vocab must build a valid tokenizer");

        let short = synthetic_sentence(10, 0);
        let medium = synthetic_sentence(50, 100);
        let long = synthetic_sentence(200, 500);
        let batch_medium: Vec<String> = (0..32)
            .map(|batch_idx| synthetic_sentence(50, 1_000 + batch_idx * 64))
            .collect();

        let short_tokens = tokenizer.tokenize(&short).real_length;
        let medium_tokens = tokenizer.tokenize(&medium).real_length;
        let long_tokens = tokenizer.tokenize(&long).real_length;
        let batch_refs: Vec<&str> = batch_medium.iter().map(String::as_str).collect();
        let batch_medium_tokens = tokenizer
            .tokenize_batch(&batch_refs)
            .iter()
            .map(|input| input.real_length)
            .sum();

        TokenizerFixture {
            tokenizer,
            short,
            medium,
            long,
            batch_medium,
            short_tokens,
            medium_tokens,
            long_tokens,
            batch_medium_tokens,
        }
    })
}

fn bench_matmul_bt(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_bt");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    let cases = [
        ("attn_proj_seq16_hidden384", 16usize, 384usize, 384usize),
        (
            "ffn_up_seq16_hidden384_intermediate1536",
            16usize,
            384usize,
            1536usize,
        ),
        (
            "ffn_down_seq16_intermediate1536_hidden384",
            16usize,
            1536usize,
            384usize,
        ),
        ("long_attn_seq128_hidden384", 128usize, 384usize, 384usize),
    ];

    for (label, m, k, n) in cases {
        let a = random_vec_with_seed(m * k, 0x1000_0000 ^ (m as u32) ^ ((k as u32) << 8));
        let b_t = random_vec_with_seed(n * k, 0x2000_0000 ^ (n as u32) ^ ((k as u32) << 8));
        let mut c_out = vec![0.0f32; m * n];
        let flops = matmul_flops(m, k, n);

        // Criterion reports elements/s here. Because the element count is configured
        // as the exact floating-point operation count per iteration, the reported
        // throughput is FLOP/s; divide by 1e9 to interpret it as GFLOPS.
        group.throughput(Throughput::Elements(flops));
        group.bench_function(BenchmarkId::new("matmul_bt", label), |b| {
            b.iter(|| {
                matmul_bt(
                    black_box(a.as_slice()),
                    black_box(b_t.as_slice()),
                    black_box(c_out.as_mut_slice()),
                    m,
                    k,
                    n,
                );
                black_box(c_out.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_layer_level_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_level_ops");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let gamma = random_layer_norm_scale(HIDDEN_SIZE, 0x3000_0001);
    let beta = random_bias_with_seed(HIDDEN_SIZE, 0x3000_0002);
    for seq_len in [16usize, 128usize] {
        let input = random_vec_with_seed(seq_len * HIDDEN_SIZE, 0x3000_1000 ^ seq_len as u32);
        group.throughput(Throughput::Elements((seq_len * HIDDEN_SIZE) as u64));
        group.bench_function(
            BenchmarkId::new("layer_norm", format!("seq{seq_len}_hidden{HIDDEN_SIZE}")),
            |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |x| {
                        layer_norm(
                            black_box(x.as_mut_slice()),
                            black_box(gamma.as_slice()),
                            black_box(beta.as_slice()),
                            HIDDEN_SIZE,
                            LAYER_NORM_EPS,
                        );
                        black_box(x.as_slice());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    for seq_len in [16usize, 128usize] {
        let input = random_vec_with_seed(seq_len * INTERMEDIATE_SIZE, 0x3000_2000 ^ seq_len as u32);
        group.throughput(Throughput::Elements((seq_len * INTERMEDIATE_SIZE) as u64));
        group.bench_function(
            BenchmarkId::new(
                "gelu",
                format!("seq{seq_len}_intermediate{INTERMEDIATE_SIZE}"),
            ),
            |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |x| {
                        gelu(black_box(x.as_mut_slice()));
                        black_box(x.as_slice());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    for seq_len in [16usize, 32usize, 64usize, 128usize] {
        let input =
            random_vec_with_seed(NUM_HEADS * seq_len * seq_len, 0x3000_3000 ^ seq_len as u32);
        group.throughput(Throughput::Elements((NUM_HEADS * seq_len * seq_len) as u64));
        group.bench_function(
            BenchmarkId::new(
                "softmax_attention",
                format!("heads{NUM_HEADS}_seq{seq_len}"),
            ),
            |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |x| {
                        softmax_attention(black_box(x.as_mut_slice()), seq_len, NUM_HEADS);
                        black_box(x.as_slice());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    for (label, dim) in [("hidden", HIDDEN_SIZE), ("intermediate", INTERMEDIATE_SIZE)] {
        let bias = random_bias_with_seed(dim, 0x3000_4000 ^ dim as u32);
        for seq_len in [16usize, 128usize] {
            let input =
                random_vec_with_seed(seq_len * dim, 0x3000_5000 ^ seq_len as u32 ^ dim as u32);
            group.throughput(Throughput::Elements((seq_len * dim) as u64));
            group.bench_function(
                BenchmarkId::new("add_bias", format!("{label}_seq{seq_len}_dim{dim}")),
                |b| {
                    b.iter_batched_ref(
                        || input.clone(),
                        |x| {
                            add_bias(black_box(x.as_mut_slice()), black_box(bias.as_slice()), dim);
                            black_box(x.as_slice());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    // Fused add_bias + gelu benchmark (compare with separate add_bias + gelu)
    {
        let bias = random_bias_with_seed(INTERMEDIATE_SIZE, 0x3000_7000);
        for seq_len in [16usize, 128usize] {
            let input =
                random_vec_with_seed(seq_len * INTERMEDIATE_SIZE, 0x3000_7001 ^ seq_len as u32);
            group.throughput(Throughput::Elements((seq_len * INTERMEDIATE_SIZE) as u64));
            group.bench_function(
                BenchmarkId::new(
                    "add_bias_gelu_fused",
                    format!("seq{seq_len}_intermediate{INTERMEDIATE_SIZE}"),
                ),
                |b| {
                    b.iter_batched_ref(
                        || input.clone(),
                        |x| {
                            add_bias_gelu(
                                black_box(x.as_mut_slice()),
                                black_box(bias.as_slice()),
                                INTERMEDIATE_SIZE,
                            );
                            black_box(x.as_slice());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // Separate add_bias + gelu for comparison
            group.bench_function(
                BenchmarkId::new(
                    "add_bias_then_gelu_separate",
                    format!("seq{seq_len}_intermediate{INTERMEDIATE_SIZE}"),
                ),
                |b| {
                    b.iter_batched_ref(
                        || input.clone(),
                        |x| {
                            add_bias(
                                black_box(x.as_mut_slice()),
                                black_box(bias.as_slice()),
                                INTERMEDIATE_SIZE,
                            );
                            gelu(black_box(x.as_mut_slice()));
                            black_box(x.as_slice());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    for seq_len in [16usize, 32usize, 64usize, 128usize] {
        let hidden_states =
            random_vec_with_seed(seq_len * HIDDEN_SIZE, 0x3000_6000 ^ seq_len as u32);
        let attention_mask = vec![1u32; seq_len];
        group.throughput(Throughput::Elements((seq_len * HIDDEN_SIZE) as u64));
        group.bench_function(
            BenchmarkId::new("mean_pool_l2_normalize", format!("seq{seq_len}")),
            |b| {
                b.iter(|| {
                    let mut pooled = mean_pool(
                        black_box(hidden_states.as_slice()),
                        black_box(attention_mask.as_slice()),
                        seq_len,
                        HIDDEN_SIZE,
                    );
                    l2_normalize(black_box(pooled.as_mut_slice()));
                    black_box(pooled);
                });
            },
        );
    }

    group.finish();
}

fn bench_tokenizer_throughput(c: &mut Criterion) {
    let fixture = tokenizer_fixture();
    let batch_refs: Vec<&str> = fixture.batch_medium.iter().map(String::as_str).collect();

    let mut group = c.benchmark_group("tokenizer_throughput");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for (label, text, token_count) in [
        (
            "single_short_10_words",
            fixture.short.as_str(),
            fixture.short_tokens,
        ),
        (
            "single_medium_50_words",
            fixture.medium.as_str(),
            fixture.medium_tokens,
        ),
        (
            "single_long_200_words",
            fixture.long.as_str(),
            fixture.long_tokens,
        ),
    ] {
        group.throughput(Throughput::Elements(token_count as u64));
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                let tokens = fixture.tokenizer.tokenize(black_box(text));
                black_box(tokens);
            });
        });
    }

    group.throughput(Throughput::Elements(fixture.batch_medium_tokens as u64));
    group.bench_function(
        BenchmarkId::from_parameter("batch_32_medium_sentences"),
        |b| {
            b.iter(|| {
                let tokens = fixture
                    .tokenizer
                    .tokenize_batch(black_box(batch_refs.as_slice()));
                black_box(tokens);
            });
        },
    );

    group.finish();
}

fn bench_full_forward_pass(c: &mut Criterion) {
    let fixture = model_fixture();
    let mut group = c.benchmark_group("full_forward_pass");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for seq_len in [8usize, 16usize, 32usize, 64usize, 128usize] {
        let input = synthetic_input(&fixture.config, seq_len, 0);
        let mut buffers = AttentionBuffers::new(
            seq_len,
            fixture.config.hidden_size,
            fixture.config.num_attention_heads,
            fixture.config.intermediate_size,
        );

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("seq_len", seq_len), |b| {
            b.iter(|| {
                let embedding = encode_with_weights(
                    &fixture.config,
                    &fixture.weights,
                    black_box(&input),
                    &mut buffers,
                );
                black_box(embedding);
            });
        });
    }

    group.finish();
}

fn bench_batch_encoding_throughput(c: &mut Criterion) {
    let fixture = model_fixture();
    let seq_len = 32usize;
    let mut group = c.benchmark_group("batch_encoding_throughput");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for batch_size in [1usize, 4usize, 8usize, 16usize, 32usize] {
        let inputs = synthetic_batch_inputs(&fixture.config, batch_size, seq_len);
        let mut buffers = AttentionBuffers::new(
            seq_len,
            fixture.config.hidden_size,
            fixture.config.num_attention_heads,
            fixture.config.intermediate_size,
        );

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_function(BenchmarkId::new("batch_size", batch_size), |b| {
            b.iter(|| {
                let outputs = encode_batch_with_weights(
                    &fixture.config,
                    &fixture.weights,
                    black_box(inputs.as_slice()),
                    &mut buffers,
                );
                black_box(outputs);
            });
        });
    }

    group.finish();
}

fn bench_attention_kernel(c: &mut Criterion) {
    let (layer_weights, _layer_storage) = synthetic_layer_weights(HIDDEN_SIZE, INTERMEDIATE_SIZE);
    let layer = &layer_weights;
    let mut group = c.benchmark_group("attention_kernel");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for seq_len in [16usize, 32usize, 64usize, 128usize] {
        let hidden_states = random_vec(seq_len * HIDDEN_SIZE);
        let attention_mask = vec![1u32; seq_len];
        let mut buffers = AttentionBuffers::new(seq_len, HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE);

        group.throughput(Throughput::Elements((NUM_HEADS * seq_len * seq_len) as u64));
        group.bench_function(BenchmarkId::new("seq_len", seq_len), |b| {
            b.iter(|| {
                let output = multi_head_attention(
                    black_box(hidden_states.as_slice()),
                    layer,
                    black_box(attention_mask.as_slice()),
                    seq_len,
                    HIDDEN_SIZE,
                    NUM_HEADS,
                    HEAD_DIM,
                    &mut buffers,
                );
                black_box(output);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_matmul_bt,
        bench_layer_level_ops,
        bench_tokenizer_throughput,
        bench_full_forward_pass,
        bench_batch_encoding_throughput,
        bench_attention_kernel
);
criterion_main!(benches);
