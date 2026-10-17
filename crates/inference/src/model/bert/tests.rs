use super::*;
use crate::forward::cpu::gelu;
use crate::pool::{cls_pool, l2_normalize, mean_pool};
use approx::assert_relative_eq;
use std::sync::Mutex;

#[test]
#[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
fn test_encode_output_shape_and_l2_norm() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");

    let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();
    let embedding = model.encode("hello world").unwrap();
    assert_eq!(embedding.len(), model.dimensions());
    let norm = (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-4);
}

/// Guards `BertModel::from_bytes` against real model files: a directory
/// with `model.safetensors` + `config.json` + `tokenizer.json` (unlike
/// `LATTICE_INFERENCE_MODEL_DIR` above, `from_bytes` needs both JSON files
/// explicitly since it has no config-from-shape inference or vocab.txt
/// fallback). Asserts the from-bytes embedding matches `from_directory`'s
/// output for the same model bit-for-bit within float tolerance.
#[test]
#[ignore = "requires LATTICE_INFERENCE_BYTES_MODEL_DIR"]
fn test_from_bytes_matches_from_directory() {
    let model_dir =
        crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_BYTES_MODEL_DIR");
    let dir = Path::new(&model_dir);

    let weights_bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let config_json = fs::read_to_string(dir.join("config.json")).unwrap();
    let tokenizer_json = fs::read_to_string(dir.join("tokenizer.json")).unwrap();

    let mut model = BertModel::from_bytes(weights_bytes, &config_json, &tokenizer_json)
        .expect("from_bytes should load a real BGE-family model directory");
    // BGE v1.5 uses CLS pooling per its model card; from_bytes has no
    // per-model pooling table (that lives in lattice-embed's
    // `EmbeddingModel::bert_pooling`), so the test sets it explicitly.
    model.set_pooling(BertPooling::CLS);

    let embedding = model.encode("hello world").unwrap();
    assert_eq!(embedding.len(), model.dimensions());
    let norm = (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-4);

    let mut reference = BertModel::from_directory(dir)
        .expect("from_directory should also load the same model directory");
    reference.set_pooling(BertPooling::CLS);
    let reference_embedding = reference.encode("hello world").unwrap();

    assert_eq!(embedding.len(), reference_embedding.len());
    for (a, b) in embedding.iter().zip(reference_embedding.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "from_bytes and from_directory embeddings diverge: {a} vs {b}"
        );
    }
}

// -------------------------------------------------------------------------
// Fused batched forward parity (mutation-sensitive)
//
// Guards the `encode_batch` rewrite: the fused batched forward path
// (`forward_batch` + `multi_head_attention_batched`) must produce the same
// per-text embeddings as looping `encode()` one text at a time. This catches
// the two correctness traps a naive batch-flattening introduces: cross-sequence
// attention leakage through the padding mask, and position-id drift for any
// sequence after the first (see forward_batch's embedding-lookup comment).
//
// Reverting `encode_batch` to loop per-item (or breaking the position-id /
// mask plumbing in `forward_batch`/`multi_head_attention_batched`) must make
// these tests fail, not just the CI bench numbers regress.
// -------------------------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let nb = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[test]
#[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
fn test_encode_batch_matches_per_item_encode_mixed_lengths() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

    // Mixed lengths (short to long) force real padding across the batch.
    let short = "hi";
    let medium = "the quick brown fox jumps over the lazy dog";
    let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod \
            tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis \
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis \
            aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat \
            nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui \
            officia deserunt mollit anim id est laborum";
    let texts: Vec<&str> = vec![
        short,
        medium,
        long_text,
        "another sentence",
        short,
        medium,
        "one more",
        long_text,
    ];

    let batched = model.encode_batch(&texts).unwrap();
    assert_eq!(batched.len(), texts.len());

    for (i, text) in texts.iter().enumerate() {
        let solo = model.encode(text).unwrap();
        let cos = cosine(&batched[i], &solo);
        let max_abs_diff = batched[i]
            .iter()
            .zip(solo.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            cos >= 0.99999,
            "text[{i}] ({text:?}...): batched vs solo cosine {cos} < 0.99999"
        );
        assert!(
            max_abs_diff <= 1e-4,
            "text[{i}] ({text:?}...): max-abs-diff {max_abs_diff} > 1e-4"
        );
    }
}

#[test]
#[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
fn test_encode_batch_boundary_max_and_min_length() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

    // One sequence at exactly the model's max position length, one at length 1
    // (single real token beyond [CLS]/[SEP] framing) -- the widest padding gap
    // this path can produce.
    let max_len_text = "word ".repeat(model.config().max_position_embeddings);
    let one_token_text = "a";
    let texts: Vec<&str> = vec![&max_len_text, one_token_text];

    let batched = model.encode_batch(&texts).unwrap();
    assert_eq!(batched.len(), 2);

    for (i, text) in texts.iter().enumerate() {
        let solo = model.encode(text).unwrap();
        let cos = cosine(&batched[i], &solo);
        let max_abs_diff = batched[i]
            .iter()
            .zip(solo.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(cos >= 0.99999, "boundary text[{i}]: cosine {cos} < 0.99999");
        assert!(
            max_abs_diff <= 1e-4,
            "boundary text[{i}]: max-abs-diff {max_abs_diff} > 1e-4"
        );
    }
}

// -------------------------------------------------------------------------
// Packed vs padded batched-encode parity (#677, mutation-sensitive)
//
// Guards the packed/varlen batched-encode rewrite: `encode_batch_packed`'s
// packed `cu_seqlens`-indexed forward pass must match `encode_batch_padded_reference`'s
// independently computed pre-#677 padded forward pass (preserved verbatim as a
// `#[cfg(test)]`-only reference; see `forward_batch_padded_reference` and
// `crate::attention::multi_head_attention_batched_padded_reference`).
//
// Both sides call `encode_batch_packed`/`encode_batch_padded_reference`
// directly, bypassing the public `encode_batch`'s single-item shortcut
// (which calls `encode()` and never touches either batched kernel), so a
// batch of size 1 genuinely exercises the packed kernel's degenerate
// one-segment `cu_seqlens = [0, total]` case instead of trivially matching
// by construction.
//
// Parametrized over four shapes; the max-variance shape ([1, 2, 128]-ish
// real tokens) is the one most likely to expose an off-by-one in the
// per-sequence `cu_seqlens[b]`/`cu_seqlens[b + 1]` window that
// `multi_head_attention_batched` indexes with -- a wrong offset there
// shows up as a large divergence on the longest sequence, not a rounding
// difference, since it would pull in (or drop) real rows from a
// neighboring sequence's attention window.
//
// NOT literal bit-for-bit equality. Measured directly on this model: any
// sequence that is the batch's longest (so the padded reference never
// actually padded it) comes back exactly bit-identical (`max_abs_diff ==
// 0.0`), but a sequence that WAS padded in the reference lands within
// ~2e-7 absolute of the packed output, not `0`. Isolated with a standalone
// harness varying only attention's total row count (real rows identical,
// extra masked padding rows appended): `multi_head_attention`'s output for
// the same real rows shifts by 1 ULP once the masked-out row count
// changes, even though `exp(-inf) == 0` zeroes every padding contribution
// algebraically. Apple's Accelerate `cblas_sgemm` (the `matmul_bt`
// backend on this platform) and the SIMD softmax reduction are not
// proven shape-invariant at the bit level for a fixed set of real
// rows plus a varying number of masked rows; only the row COUNT changes,
// but that changes GEMM tiling/blocking and reduction order, which
// reassociates the sum. This is the same class of accepted f32
// reassociation already documented elsewhere in this crate (e.g. the
// LoRA adapter reorder in `forward_with_hook`), just measured here to a
// much tighter bound than that convention's cosine/max-abs-diff check
// because this parity claim is far stronger (identical kernel, only the
// row count and per-sequence real length differ). `TOLERANCE` below is
// ~45x the largest diff measured across all four shapes on this model,
// and still 10x tighter than the crate's existing batched-vs-solo parity
// tolerance (1e-4) -- a genuine `cu_seqlens` indexing bug pulls in a
// neighboring sequence's rows and blows through it by several orders of
// magnitude, not a rounding-scale amount; that class stays caught.
// -------------------------------------------------------------------------

#[test]
#[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
fn encode_batch_packed_matches_padded() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

    let short = "hi";
    let medium = "the quick brown fox jumps over the lazy dog";
    let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod \
            tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis \
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis \
            aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat \
            nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui \
            officia deserunt mollit anim id est laborum";
    let uniform_64 = "word ".repeat(64);
    let one_word = "a";
    let two_words = "hi there";
    let long_128 = "word ".repeat(128);

    let shapes: Vec<(&str, Vec<&str>)> = vec![
        ("single_row", vec![medium]),
        (
            "uniform_three_same_length",
            vec![
                uniform_64.as_str(),
                uniform_64.as_str(),
                uniform_64.as_str(),
            ],
        ),
        (
            "max_variance_1_2_128",
            vec![one_word, two_words, long_128.as_str()],
        ),
        ("mixed_5_30_100", vec![short, medium, long_text]),
    ];

    // ~45x the largest max-abs-diff measured across all four shapes on this
    // model (2.2e-7), and 10x tighter than the crate's existing batched-vs-solo
    // parity tolerance (1e-4, see `test_encode_batch_matches_per_item_encode_mixed_lengths`).
    const TOLERANCE: f32 = 1e-5;

    for (name, texts) in shapes {
        let packed = model.encode_batch_packed(&texts).unwrap();
        let padded = model.encode_batch_padded_reference(&texts);

        assert_eq!(packed.len(), padded.len(), "{name}: output count mismatch");
        for (i, (p, q)) in packed.iter().zip(padded.iter()).enumerate() {
            assert_eq!(
                p.len(),
                q.len(),
                "{name}: text[{i}] embedding length mismatch"
            );
            let max_abs_diff = p
                .iter()
                .zip(q.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs_diff <= TOLERANCE,
                "{name}: text[{i}] packed vs padded max_abs_diff {max_abs_diff} exceeds {TOLERANCE}"
            );
        }
    }
}

// -------------------------------------------------------------------------
// FFN-intermediate LoRA-adapter reorder correctness (#675, mutation-sensitive)
//
// `forward_with_hook` calls `apply_ffn_intermediate_lora_and_activation`,
// which runs `lora.apply("ffn_intermediate", ...)` before the fused
// `add_bias_gelu` kernel, on the argument that the bias-add and the
// adapter delta are both additions into the same buffer ahead of a single
// GELU, so their relative order does not matter as long as both land
// before GELU (see that function's and `forward_with_hook`'s doc comments
// for the accepted-reassociation argument).
//
// Two tests cover this, both of which must fail if the ordering regresses
// to add_bias_gelu-then-adapter (delta landing after GELU instead of
// before):
//   - `test_ffn_intermediate_stage_lora_delta_lands_before_gelu` runs by
//     default (no `#[ignore]`): it calls
//     `apply_ffn_intermediate_lora_and_activation` directly against small
//     synthetic buffers, so it needs no downloaded model and always runs
//     in CI.
//   - `test_ffn_intermediate_lora_delta_lands_before_gelu` is the
//     end-to-end version through a real `BertModel` and a real adapter
//     hook reachable the way `CrossEncoderModel::score_with_hook` reaches
//     it; it is `#[ignore]`d and gated on `LATTICE_INFERENCE_MODEL_DIR`
//     because building a real BERT-family model in-process needs a
//     downloaded checkpoint (safetensors weights, config, and tokenizer),
//     which this crate does not synthesize for tests.
// -------------------------------------------------------------------------

/// Default (non-`#[ignore]`) unit test: exercises
/// `apply_ffn_intermediate_lora_and_activation` directly against small,
/// fixed synthetic buffers, with no model required, so this runs in CI
/// by default.
#[test]
fn test_ffn_intermediate_stage_lora_delta_lands_before_gelu() {
    // Small, fixed sizes and values -- deterministic, no RNG dependency.
    let seq_len = 2;
    let hidden_size = 3;
    let intermediate_size = 4;
    let used_hidden = seq_len * hidden_size;
    let used_intermediate = seq_len * intermediate_size;

    let hidden: Vec<f32> = (0..used_hidden).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let weight: Vec<f32> = (0..intermediate_size * hidden_size)
        .map(|i| 0.05 * (i as f32 + 1.0))
        .collect();
    let bias: Vec<f32> = (0..intermediate_size)
        .map(|i| 0.01 * (i as f32 + 1.0))
        .collect();
    let delta = 0.05_f32;

    let mut raw = vec![0.0f32; used_intermediate];
    matmul_bt(
        &hidden,
        &weight,
        &mut raw,
        seq_len,
        hidden_size,
        intermediate_size,
    );

    // Production composition under test: the same function
    // `forward_with_hook` calls.
    let hook = CapturingDeltaHook {
        target_module: "ffn_intermediate",
        delta,
        captured_x: Mutex::new(Vec::new()),
    };
    let mut produced = raw.clone();
    apply_ffn_intermediate_lora_and_activation(
        &hook,
        0,
        &hidden,
        &mut produced,
        &bias,
        hidden_size,
        intermediate_size,
    );

    // (a) The adapter must actually reach the result.
    let mut without_delta = raw.clone();
    add_bias(&mut without_delta, &bias, intermediate_size);
    gelu(&mut without_delta);
    let max_abs_diff_vs_no_delta = produced
        .iter()
        .zip(without_delta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff_vs_no_delta > 1e-3,
        "an active ffn_intermediate adapter must move the output; \
             max-abs-diff vs no-delta was only {max_abs_diff_vs_no_delta}"
    );

    // (b) Independent reference that hardcodes "delta lands before gelu"
    // via separate, unfused matmul_bt/add_bias/gelu calls.
    let mut reference = raw;
    add_bias(&mut reference, &bias, intermediate_size);
    for v in reference.iter_mut() {
        *v += delta;
    }
    gelu(&mut reference); // separate, unfused gelu -- independent of add_bias_gelu

    let max_abs_diff_vs_reference = produced
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff_vs_reference < 1e-5,
        "apply_ffn_intermediate_lora_and_activation does not match the \
             delta-lands-before-gelu reference: max-abs-diff {max_abs_diff_vs_reference}"
    );
}

/// `LoraHook` that captures the FFN block's input activation (`x`) for one
/// target module and adds a constant delta to that projection's output.
///
/// Capturing `x` rather than `output` keeps the reference computation in
/// the test below independent of where the hook actually fires relative
/// to bias/GELU: `x` is produced before the FFN block starts, so its
/// value does not depend on the ordering under test.
struct CapturingDeltaHook {
    target_module: &'static str,
    delta: f32,
    captured_x: Mutex<Vec<f32>>,
}

impl LoraHook for CapturingDeltaHook {
    fn apply(&self, _layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
        if module != self.target_module {
            return;
        }
        self.captured_x.lock().unwrap().extend_from_slice(x);
        for o in output.iter_mut() {
            *o += self.delta;
        }
    }
}

/// Build a `TokenizedInput` + fresh `AttentionBuffers` for one text, the
/// same way `BertModel::encode` does internally.
fn tokenize_and_buffers(
    model: &BertModel,
    text: &str,
) -> (crate::tokenizer::TokenizedInput, AttentionBuffers) {
    let input = model.tokenizer.tokenize(text);
    let seq_len = input.real_length;
    let buffers = AttentionBuffers::new(
        seq_len,
        model.config.hidden_size,
        model.config.num_attention_heads,
        model.config.intermediate_size,
    );
    (input, buffers)
}

#[test]
#[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
fn test_ffn_intermediate_lora_delta_lands_before_gelu() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let mut model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

    // Truncate to a single transformer layer so the adapter's effect does
    // not cascade through further layers: the reference below is then a
    // direct, checkable reconstruction of exactly one FFN block.
    model.weights.layers.truncate(1);
    model.fused_qkv.truncate(1);
    model.config.num_hidden_layers = 1;

    let text = "the quick brown fox jumps over the lazy dog";
    let delta = 0.05_f32;

    // Baseline: no adapter.
    let (input, mut buffers) = tokenize_and_buffers(&model, text);
    let baseline = model.forward_tokenized_with_hook(&input, &mut buffers, &NoopLoraHook);

    // With adapter: a constant delta added to the ffn_intermediate
    // projection, capturing its input activation as it flows through.
    let hook = CapturingDeltaHook {
        target_module: "ffn_intermediate",
        delta,
        captured_x: Mutex::new(Vec::new()),
    };
    let (input2, mut buffers2) = tokenize_and_buffers(&model, text);
    let with_delta = model.forward_tokenized_with_hook(&input2, &mut buffers2, &hook);

    // (a) The adapter must actually reach the result -- otherwise this
    // whole path is dead and the rest of the test proves nothing.
    let max_abs_diff_vs_baseline = with_delta
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff_vs_baseline > 1e-3,
        "an active ffn_intermediate adapter must move the output; \
             max-abs-diff vs no-op baseline was only {max_abs_diff_vs_baseline}"
    );

    // (b) Build a reference that hardcodes "delta lands before GELU"
    // using separate, unfused kernel calls (matmul_bt / add_bias / gelu
    // one at a time), so it does not just replay whatever internal order
    // forward_with_hook happens to use.
    let seq_len = input2.real_length;
    let hidden_size = model.config.hidden_size;
    let intermediate_size = model.config.intermediate_size;
    let used_hidden = seq_len * hidden_size;
    let used_intermediate = seq_len * intermediate_size;
    let layer = &model.weights.layers[0];

    let captured_x = hook.captured_x.lock().unwrap().clone();
    assert_eq!(captured_x.len(), used_hidden);

    let mut raw = vec![0.0f32; used_intermediate];
    matmul_bt(
        &captured_x,
        layer.ffn_intermediate_weight.data,
        &mut raw,
        seq_len,
        hidden_size,
        intermediate_size,
    );
    add_bias(
        &mut raw,
        layer.ffn_intermediate_bias.data,
        intermediate_size,
    );
    for v in raw.iter_mut() {
        *v += delta;
    }
    gelu(&mut raw); // separate, unfused gelu -- independent of add_bias_gelu

    let mut reference = vec![0.0f32; used_hidden];
    matmul_bt(
        &raw,
        layer.ffn_output_weight.data,
        &mut reference,
        seq_len,
        intermediate_size,
        hidden_size,
    );
    add_bias(&mut reference, layer.ffn_output_bias.data, hidden_size);
    for i in 0..used_hidden {
        reference[i] += captured_x[i];
    }
    layer_norm(
        &mut reference,
        layer.ffn_layer_norm_weight.data,
        layer.ffn_layer_norm_bias.data,
        hidden_size,
        model.config.layer_norm_eps,
    );

    assert_eq!(reference.len(), with_delta.len());
    let max_abs_diff_vs_reference = reference
        .iter()
        .zip(with_delta.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff_vs_reference < 1e-4,
        "forward_with_hook's ffn_intermediate output does not match the \
             delta-lands-before-gelu reference: max-abs-diff {max_abs_diff_vs_reference}"
    );
}

// -------------------------------------------------------------------------
// Deterministic pooling tests (P1-E3)
//
// These tests use fixed hidden-state tensors — no model weights needed.
// They validate the pooling routing at the kernel level: CLS extracts
// position 0, mean computes an attention-mask-weighted average, and L2
// normalisation produces a unit vector in both cases.
// -------------------------------------------------------------------------

/// Fixed 2-token, 4-dim hidden-state tensor.
///
/// Token 0 (CLS):  [1.0, 0.0, 0.0, 0.0]
/// Token 1 (word): [0.0, 1.0, 0.0, 0.0]
/// Both tokens are real (attention_mask = [1, 1]).
fn hidden_2x4() -> (Vec<f32>, Vec<u32>) {
    let hidden = vec![
        1.0_f32, 0.0, 0.0, 0.0, // token 0 (CLS)
        0.0_f32, 1.0, 0.0, 0.0, // token 1 (word)
    ];
    let mask = vec![1_u32, 1];
    (hidden, mask)
}

/// CLS pooling returns the first-token hidden state ([1,0,0,0]), then L2 normalises.
///
/// The CLS token is already unit-length here, so after L2 it stays [1,0,0,0].
/// This matches the BGE model-card recipe: `model_output[0][:, 0]` + L2.
#[test]
fn test_cls_pool_extracts_first_token_and_l2_unit_norm() {
    let (hidden, _mask) = hidden_2x4();
    let seq_len = 2;
    let hidden_size = 4;

    let mut pooled = cls_pool(&hidden, seq_len, hidden_size);

    // Before L2: should be the CLS row [1,0,0,0].
    assert_eq!(
        pooled,
        vec![1.0, 0.0, 0.0, 0.0],
        "CLS row mismatch before L2"
    );

    l2_normalize(&mut pooled);

    // CLS row is already unit-length → unchanged.
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    assert_relative_eq!(pooled[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(pooled[1], 0.0, epsilon = 1e-6);
}

/// Mean pooling with uniform mask averages all tokens, then L2 normalises.
///
/// With hidden = [[1,0,0,0],[0,1,0,0]] and mask [1,1],
/// mean = [0.5, 0.5, 0, 0].  After L2: [1/√2, 1/√2, 0, 0] ≈ [0.7071, 0.7071, 0, 0].
///
/// This matches the E5/MiniLM model-card recipe: masked mean pooling + L2.
#[test]
fn test_mean_pool_averages_masked_tokens_and_l2_unit_norm() {
    let (hidden, mask) = hidden_2x4();
    let seq_len = 2;
    let hidden_size = 4;

    let mut pooled = mean_pool(&hidden, &mask, seq_len, hidden_size);

    // Before L2: mean of [1,0,0,0] and [0,1,0,0] = [0.5, 0.5, 0, 0].
    assert_relative_eq!(pooled[0], 0.5, epsilon = 1e-6);
    assert_relative_eq!(pooled[1], 0.5, epsilon = 1e-6);
    assert_relative_eq!(pooled[2], 0.0, epsilon = 1e-6);
    assert_relative_eq!(pooled[3], 0.0, epsilon = 1e-6);

    l2_normalize(&mut pooled);

    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-6);

    // L2 of [0.5, 0.5, 0, 0]: magnitude = √0.5, so normalised = [1/√2, 1/√2, 0, 0].
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    assert_relative_eq!(pooled[0], inv_sqrt2, epsilon = 1e-5);
    assert_relative_eq!(pooled[1], inv_sqrt2, epsilon = 1e-5);
}

/// CLS and mean pooling of the same hidden states produce DIFFERENT vectors.
///
/// This is the key correctness guarantee for P1-E3: using the wrong pooling
/// strategy for a model produces a meaningfully different embedding.
#[test]
fn test_cls_and_mean_produce_different_embeddings() {
    let (hidden, mask) = hidden_2x4();
    let seq_len = 2;
    let hidden_size = 4;

    let mut cls = cls_pool(&hidden, seq_len, hidden_size);
    let mut mean = mean_pool(&hidden, &mask, seq_len, hidden_size);

    l2_normalize(&mut cls);
    l2_normalize(&mut mean);

    // CLS = [1, 0, 0, 0],  mean = [1/√2, 1/√2, 0, 0] — these differ.
    assert_ne!(
        cls, mean,
        "CLS and mean pooling must produce different unit vectors"
    );
}

/// Mean pooling with a padding mask ignores masked positions.
///
/// With hidden = [[1,0,0,0],[0,1,0,0]] and mask [1, 0],
/// only token 0 contributes: mean = [1, 0, 0, 0].
#[test]
fn test_mean_pool_respects_padding_mask() {
    let hidden = vec![
        1.0_f32, 0.0, 0.0, 0.0, // token 0 (real)
        0.0_f32, 1.0, 0.0, 0.0, // token 1 (pad, mask=0)
    ];
    let mask = vec![1_u32, 0]; // second token is padding
    let seq_len = 2;
    let hidden_size = 4;

    let pooled = mean_pool(&hidden, &mask, seq_len, hidden_size);

    // Only token 0 is unmasked → mean = [1,0,0,0].
    assert_relative_eq!(pooled[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(pooled[1], 0.0, epsilon = 1e-6);
}
