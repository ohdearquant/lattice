//! An over-complete LoRA factorization must score through the public
//! cross-encoder boundary.
//!
//! `rank > min(d_in, d_out)` makes `B @ A` a redundant factorization — the
//! same update is exactly representable at the lower rank — but it is still
//! a well-defined update, and `blend_lora_adapters` produces exactly this
//! shape because the blended rank is the sum of the source ranks.
//! `CrossEncoderModel::score_with_hook` validates the adapter through
//! `LoraHook::validate_against_bert` and maps any failure to
//! `InferenceError::InvalidInput`, so a rank ceiling in the validator turns
//! a correct adapter into a hard error at the public scoring API. This test
//! drives a real `LoraAdapter` through that API against a tiny synthetic
//! BERT cross-encoder checkpoint (no network access, no downloaded model).
//!
//! It also pins the behaviour of the other arm of that delegation: a hook
//! that does *not* override `validate_against_bert` and declares a geometry
//! the model disagrees with.

#![cfg(feature = "inference-hook")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lattice_inference::lora_hook::LoraHook;
use lattice_inference::model::CrossEncoderModel;
use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer, apply_lora};

const HIDDEN_SIZE: usize = 4;
const INTERMEDIATE_SIZE: usize = 4;
const NUM_HIDDEN_LAYERS: usize = 1;
const NUM_ATTENTION_HEADS: usize = 1;
const MAX_POSITION_EMBEDDINGS: usize = 32;
const TYPE_VOCAB_SIZE: usize = 2;
// Must match the tokenizer fixture's vocab size: `word_embeddings` is indexed
// directly by token id, so an undersized table would panic for reasons
// unrelated to what this test targets.
const VOCAB_SIZE: usize = 30522;

/// The cross-encoder requires a tokenizer with BERT pair tokenization. Read
/// the existing WordPiece fixture rather than vendoring a second 700 KB copy
/// of it; a runtime read keeps the failure legible if the path ever moves.
fn tokenizer_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inference/tests/fixtures/tokenizers/bge-small-en-v1.5/tokenizer.json")
}

fn write_f32_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut entries: Vec<String> = Vec::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            values.len(),
            expected_len,
            "tensor {name} shape/value length mismatch"
        );
        let start = payload.len();
        for &v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        let shape_json = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape_json}],"data_offsets":[{start},{end}]}}"#
        ));
    }

    let header = format!("{{{}}}", entries.join(","));
    let header_bytes = header.into_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&payload);

    std::fs::write(path, out).unwrap();
}

/// Deterministic, small, finite fill values — the numeric output of the
/// forward pass is irrelevant here; only "panicked vs. Err vs. Ok" matters.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.01 * (((i + seed) % 7) as f32 - 3.0))
        .collect()
}

fn build_synthetic_cross_encoder_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    let tokenizer_path = tokenizer_fixture_path();
    let tokenizer_json = std::fs::read_to_string(&tokenizer_path).unwrap_or_else(|e| {
        panic!(
            "tokenizer fixture {} unreadable: {e}",
            tokenizer_path.display()
        )
    });
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer_json).unwrap();

    let config_json = format!(
        r#"{{
            "vocab_size": {VOCAB_SIZE},
            "hidden_size": {HIDDEN_SIZE},
            "num_hidden_layers": {NUM_HIDDEN_LAYERS},
            "num_attention_heads": {NUM_ATTENTION_HEADS},
            "intermediate_size": {INTERMEDIATE_SIZE},
            "max_position_embeddings": {MAX_POSITION_EMBEDDINGS},
            "type_vocab_size": {TYPE_VOCAB_SIZE},
            "layer_norm_eps": 1e-5
        }}"#
    );
    std::fs::write(dir.path().join("config.json"), config_json).unwrap();

    let h = HIDDEN_SIZE;
    let i = INTERMEDIATE_SIZE;
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        (
            "embeddings.word_embeddings.weight",
            vec![VOCAB_SIZE, h],
            fill(VOCAB_SIZE * h, 1),
        ),
        (
            "embeddings.position_embeddings.weight",
            vec![MAX_POSITION_EMBEDDINGS, h],
            fill(MAX_POSITION_EMBEDDINGS * h, 2),
        ),
        (
            "embeddings.token_type_embeddings.weight",
            vec![TYPE_VOCAB_SIZE, h],
            fill(TYPE_VOCAB_SIZE * h, 3),
        ),
        ("embeddings.LayerNorm.weight", vec![h], vec![1.0; h]),
        ("embeddings.LayerNorm.bias", vec![h], vec![0.0; h]),
        (
            "encoder.layer.0.attention.self.query.weight",
            vec![h, h],
            fill(h * h, 4),
        ),
        (
            "encoder.layer.0.attention.self.query.bias",
            vec![h],
            fill(h, 5),
        ),
        (
            "encoder.layer.0.attention.self.key.weight",
            vec![h, h],
            fill(h * h, 6),
        ),
        (
            "encoder.layer.0.attention.self.key.bias",
            vec![h],
            fill(h, 7),
        ),
        (
            "encoder.layer.0.attention.self.value.weight",
            vec![h, h],
            fill(h * h, 8),
        ),
        (
            "encoder.layer.0.attention.self.value.bias",
            vec![h],
            fill(h, 9),
        ),
        (
            "encoder.layer.0.attention.output.dense.weight",
            vec![h, h],
            fill(h * h, 10),
        ),
        (
            "encoder.layer.0.attention.output.dense.bias",
            vec![h],
            fill(h, 11),
        ),
        (
            "encoder.layer.0.attention.output.LayerNorm.weight",
            vec![h],
            vec![1.0; h],
        ),
        (
            "encoder.layer.0.attention.output.LayerNorm.bias",
            vec![h],
            vec![0.0; h],
        ),
        (
            "encoder.layer.0.intermediate.dense.weight",
            vec![i, h],
            fill(i * h, 12),
        ),
        (
            "encoder.layer.0.intermediate.dense.bias",
            vec![i],
            fill(i, 13),
        ),
        (
            "encoder.layer.0.output.dense.weight",
            vec![h, i],
            fill(h * i, 14),
        ),
        ("encoder.layer.0.output.dense.bias", vec![h], fill(h, 15)),
        (
            "encoder.layer.0.output.LayerNorm.weight",
            vec![h],
            vec![1.0; h],
        ),
        (
            "encoder.layer.0.output.LayerNorm.bias",
            vec![h],
            vec![0.0; h],
        ),
        ("classifier.weight", vec![1, h], fill(h, 16)),
        ("classifier.bias", vec![1], vec![0.1]),
    ];
    write_f32_safetensors(&dir.path().join("model.safetensors"), &tensors);

    dir
}

/// `rank` one above `min(d_in, d_out)`, with buffers consistent with that
/// declared rank. Every offset `apply_lora` computes stays inside the
/// buffers; the only thing unusual about this adapter is that its
/// factorization is redundant.
fn over_complete_query_adapter() -> LoraAdapter {
    let rank = HIDDEN_SIZE + 1;
    let mut layers = HashMap::new();
    layers.insert(
        (0, "query".to_string()),
        LoraLayer {
            a: fill(rank * HIDDEN_SIZE, 17),
            b: fill(HIDDEN_SIZE * rank, 18),
            d_in: HIDDEN_SIZE,
            d_out: HIDDEN_SIZE,
            rank,
        },
    );
    LoraAdapter::new(
        LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["query".to_string()],
        },
        layers,
    )
    .expect("buffers match the declared rank")
}

#[test]
fn cross_encoder_scores_over_complete_lora_adapter() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let adapter = over_complete_query_adapter();

    let score = model
        .score_with_hook("what is rust", "rust is a language", &adapter)
        .expect("an over-complete but correctly shaped adapter must score, not error");
    assert!(
        (0.0..=1.0).contains(&score),
        "score {score} out of sigmoid range"
    );
}

/// The batch entry point validates per document through the same
/// `score_with_hook`; cover it so the ceiling cannot be reintroduced behind
/// a wrapper that this suite does not exercise.
#[test]
fn cross_encoder_batch_scores_over_complete_lora_adapter() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();
    let adapter = over_complete_query_adapter();

    let scores = model
        .score_batch_with_hook(
            "what is rust",
            &["rust is a language", "ferris is a crab"],
            &adapter,
        )
        .expect("an over-complete but correctly shaped adapter must score in batch too");
    assert_eq!(scores.len(), 2);
    assert!(
        scores.iter().all(|s| (0.0..=1.0).contains(s)),
        "scores {scores:?} out of sigmoid range"
    );
}

/// A hook that applies a real `LoraLayer` through `apply_lora` but leaves
/// `LoraHook::validate_against_bert` at its trait default, so the geometry
/// check `score_with_hook` delegates to accepts it unconditionally. This is
/// the shape any out-of-workspace implementor gets for free.
struct UnvalidatedHook {
    layer: LoraLayer,
    module: &'static str,
    scale: f32,
}

impl LoraHook for UnvalidatedHook {
    fn apply(&self, _layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
        if module != self.module {
            return;
        }
        apply_lora(&self.layer, self.scale, x, output);
    }
}

/// Targets `"ffn_output"`, whose per-row output buffer is `hidden_size`
/// wide. Buffers are internally consistent with the declared `d_out`, so the
/// only disagreement is between the adapter and the model.
fn unvalidated_ffn_output_hook(d_out: usize) -> UnvalidatedHook {
    let rank = 2;
    UnvalidatedHook {
        layer: LoraLayer {
            a: fill(rank * INTERMEDIATE_SIZE, 19),
            b: fill(d_out * rank, 20),
            d_in: INTERMEDIATE_SIZE,
            d_out,
            rank,
        },
        module: "ffn_output",
        // A LayerNorm follows the hooked projection, so a unit-scaled delta
        // built from `fill` values moves the final score by ~1e-7 — inside
        // f32 noise. Scale it up so "applied" and "not applied" are
        // unambiguously distinguishable.
        scale: 64.0,
    }
}

/// A hook reaching the forward pass without applying anything — the
/// reference score for "was the adapter applied at all?", taken through the
/// same hooked forward path so no unhooked/hooked path difference can be
/// mistaken for an applied update.
struct NeutralHook;

impl LoraHook for NeutralHook {
    fn apply(&self, _layer_idx: usize, _module: &str, _x: &[f32], _output: &mut [f32]) {}
}

/// A non-overriding hook declaring `d_out > hidden_size` scores without
/// panicking, and its update is not applied: `apply_lora`'s shape check is an
/// exact-width match (`output.len() == d_out`), which this hook fails, so it
/// returns early and the projection row is left exactly as the base weights
/// produced it.
///
/// The predicate is exact deliberately. The inequality it replaced —
/// `output.len() >= d_out` — passed for an under-declared `d_out`, and the
/// accumulate loop then wrote an update across a prefix of the row while the
/// rest kept base weights: no panic, no error, a plausible wrong score. Do not
/// restore the inequality to make this test read more naturally.
#[test]
fn cross_encoder_drops_the_update_of_an_unvalidated_oversized_hook() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let reference = model
        .score_with_hook("what is rust", "rust is a language", &NeutralHook)
        .expect("a hook that applies nothing must score");
    let oversized = model
        .score_with_hook(
            "what is rust",
            "rust is a language",
            &unvalidated_ffn_output_hook(HIDDEN_SIZE + 1),
        )
        .expect("an unvalidated oversized hook must not error at this boundary");

    assert_eq!(
        oversized, reference,
        "an oversized d_out must leave the projection untouched, not apply a \
         partial update"
    );
}

/// The other direction of the same mismatch, which used to behave differently
/// and deliberately no longer does.
///
/// `apply_lora`'s shape check was `output.len() >= d_out`, an inequality, while
/// the accumulate loop writes `output[..d_out]`. An UNDER-declared `d_out`
/// therefore passed the check and wrote a partial row: an update computed for a
/// geometry the model does not have across the prefix, base weights across the
/// rest, no panic, no error, and a finite in-range score. That is a silent
/// wrong answer, which is a worse failure than the dropped update the
/// over-declared direction produces.
///
/// The check is now exact width, so both directions no-op. Nothing in this
/// workspace declares a `d_out` other than the projection's own width — both
/// validators reject any dims disagreement, and the engine hands `apply_lora`
/// `chunks_exact_mut` rows — so tightening costs no legitimate caller anything.
///
/// This test is the guard on that: it fails if the inequality ever comes back.
#[test]
fn cross_encoder_drops_the_update_of_an_unvalidated_undersized_hook() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let reference = model
        .score_with_hook("what is rust", "rust is a language", &NeutralHook)
        .expect("a hook that applies nothing must score");
    let undersized = model
        .score_with_hook(
            "what is rust",
            "rust is a language",
            &unvalidated_ffn_output_hook(HIDDEN_SIZE - 1),
        )
        .expect("an unvalidated undersized hook must not error at this boundary");

    assert_eq!(
        undersized, reference,
        "an undersized d_out must leave the projection untouched; a score that \
         differs from the reference means a partial prefix update was applied, \
         which is the silent-wrong-score hole the exact-width check closes"
    );
}

/// An adapter targeting only attention projections reaches the forward path.
///
/// The BERT forward path dispatches all six BERT projections to the hook, not
/// only the two FFN ones: `bert.rs` routes attention through
/// `attention::standard`, which asks the hook for `query`, `key`, `value` and
/// `attn_output`. An adapter built solely from attention layers is therefore
/// applied, not silently inert.
///
/// The adapter here is deliberately strong. The neighbouring
/// `over_complete_query_adapter` uses `fill`, whose values are bounded by
/// 0.03, so its update reaches the score at roughly 1e-8 — below f32
/// resolution at a sigmoid output near 0.5. A test built on that adapter
/// reports "no change" whether the update is applied or dropped, which is the
/// same as having no test. Magnitude is part of what makes this one able to
/// fail.
#[test]
fn cross_encoder_applies_an_attention_only_adapter() {
    let dir = build_synthetic_cross_encoder_dir();
    let model = CrossEncoderModel::from_directory(dir.path()).unwrap();

    let rank = 2usize;
    let mut layers = HashMap::new();
    layers.insert(
        (0, "value".to_string()),
        LoraLayer {
            // Deliberately non-uniform. A uniform `a` makes every row of
            // `A @ x` proportional to `sum(x)`, and the preceding LayerNorm
            // zero-means each row, so a uniform adapter computes an update of
            // approximately zero and the test would report "inert" for an
            // adapter that is in fact applied correctly.
            a: (0..rank * HIDDEN_SIZE)
                .map(|i| if i % 2 == 0 { 0.9 } else { -0.4 })
                .collect(),
            b: (0..HIDDEN_SIZE * rank)
                .map(|i| if i % 3 == 0 { 0.8 } else { -0.6 })
                .collect(),
            d_in: HIDDEN_SIZE,
            d_out: HIDDEN_SIZE,
            rank,
        },
    );
    let adapter = LoraAdapter::new(
        LoraConfig {
            rank,
            alpha: rank as f32,
            target_modules: vec!["value".to_string()],
        },
        layers,
    )
    .expect("buffers match the declared rank");

    let reference = model
        .score_with_hook("what is rust", "rust is a language", &NeutralHook)
        .expect("a hook that applies nothing must score");
    let adapted = model
        .score_with_hook("what is rust", "rust is a language", &adapter)
        .expect("an attention-only adapter must score");

    assert_ne!(
        adapted, reference,
        "an adapter targeting only `value` changed nothing: the attention \
         projections are not reaching the hook"
    );
}
