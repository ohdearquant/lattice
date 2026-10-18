use super::*;
use crate::model::ernie45::{Ernie45LayerWeights, Ernie45RopeScaling};
use crate::vision::paddleocr_vit::PaddleOcrVitLayerWeights;

fn tokenizer() -> GemmaBpeTokenizer {
    GemmaBpeTokenizer::from_ernie_tokenizer_json(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/paddleocr_vl/tokenizer/tokenizer.json"),
    )
    .expect("committed ERNIE tokenizer")
}

fn rectangular_diagonal(rows: usize, columns: usize, value: f32) -> Vec<f32> {
    let mut weights = vec![0.0; rows * columns];
    for row in 0..rows {
        weights[row * columns + row % columns] = value;
    }
    weights
}

fn model(winner: u32) -> PaddleOcrVlModel {
    let h = 8;
    let inter = 12;
    let q = 12;
    let kv = 6;
    let vocab = 103_424;
    let decoder_cfg = Ernie45Config {
        hidden_size: h,
        intermediate_size: inter,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 6,
        vocab_size: vocab,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_scaling: Ernie45RopeScaling {
            mrope_section: vec![1, 1, 1],
        },
        tie_word_embeddings: false,
        use_bias: false,
    };
    let mut lm_head = vec![0.0; vocab * h];
    lm_head[winner as usize * h..][..h].fill(1.0);
    let decoder = Ernie45Model::new(
        decoder_cfg.clone(),
        Ernie45Weights {
            embed_tokens: (0..vocab * h)
                .map(|index| 1.0 + (index % h) as f32 / 32.0)
                .collect(),
            layers: vec![Ernie45LayerWeights {
                q_proj: rectangular_diagonal(q, h, 0.125),
                k_proj: rectangular_diagonal(kv, h, 0.125),
                v_proj: rectangular_diagonal(kv, h, 0.125),
                o_proj: rectangular_diagonal(h, q, 0.125),
                gate_proj: rectangular_diagonal(inter, h, 0.125),
                up_proj: rectangular_diagonal(inter, h, 0.125),
                down_proj: rectangular_diagonal(h, inter, 0.125),
                input_layernorm: vec![1.0; h],
                post_attention_layernorm: vec![1.0; h],
            }],
            final_norm: vec![1.0; h],
            lm_head,
        },
    )
    .expect("nonzero GQA decoder");
    let vision_cfg = PaddleOcrVisionConfig {
        hidden_size: h,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_channels: 3,
        patch_size: 2,
        image_size: 4,
        layer_norm_eps: 1e-6,
        spatial_merge_size: 2,
        text_hidden_size: h,
    };
    vision_cfg.validate().expect("tiny vision geometry");
    let merged = h * 4;
    let vision_weights = PaddleOcrVisionWeights {
        patch_weight: rectangular_diagonal(h, 12, 0.125),
        patch_bias: vec![0.0; h],
        pos_embed: (0..4 * h).map(|i| (i % h) as f32 / 32.0).collect(),
        layers: vec![PaddleOcrVitLayerWeights {
            ln1_weight: vec![1.0; h],
            ln1_bias: vec![0.0; h],
            q_weight: rectangular_diagonal(h, h, 0.125),
            q_bias: vec![0.0; h],
            k_weight: rectangular_diagonal(h, h, 0.125),
            k_bias: vec![0.0; h],
            v_weight: rectangular_diagonal(h, h, 0.125),
            v_bias: vec![0.0; h],
            out_weight: rectangular_diagonal(h, h, 0.125),
            out_bias: vec![0.0; h],
            ln2_weight: vec![1.0; h],
            ln2_bias: vec![0.0; h],
            fc1_weight: rectangular_diagonal(16, h, 0.125),
            fc1_bias: vec![0.0; 16],
            fc2_weight: rectangular_diagonal(h, 16, 0.125),
            fc2_bias: vec![0.0; h],
        }],
        post_ln_weight: vec![1.0; h],
        post_ln_bias: vec![0.0; h],
        proj_norm_weight: vec![1.0; h],
        proj_norm_bias: vec![0.0; h],
        proj_l1_weight: rectangular_diagonal(merged, merged, 0.125),
        proj_l1_bias: vec![0.0; merged],
        proj_l2_weight: rectangular_diagonal(h, merged, 0.125),
        proj_l2_bias: vec![1.0; h],
    };
    PaddleOcrVlModel {
        decoder,
        decoder_cfg,
        vision_cfg,
        vision_weights,
        processor_cfg: PaddleOcrImageProcessorConfig {
            min_pixels: 16,
            max_pixels: 16,
            patch_size: 2,
            merge_size: 2,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
        },
    }
}

fn image() -> Vec<u8> {
    (0..48).map(|i| ((i * 37 + 11) % 256) as u8).collect()
}

#[test]
fn cached_request_reuses_all_nineteen_buffers() {
    let model = model(7);
    let tokenizer = tokenizer();
    let rgb = image();
    let mut storage = None;
    let mut calls = 0;
    let actual = model
        .generate_greedy_observed(&tokenizer, &rgb, 4, 4, "OCR:", 5, |scratch| {
            calls += 1;
            assert_eq!(scratch.test_decode_calls(), calls);
            let current = scratch.test_storage_snapshot();
            assert_eq!(current.len(), 19);
            assert!(
                current
                    .iter()
                    .all(|&(pointer, capacity)| pointer != 0 && capacity > 0)
            );
            if let Some(previous) = &storage {
                assert_eq!(&current, previous, "request scratch was replaced");
            } else {
                storage = Some(current);
            }
        })
        .expect("actual image-to-cached-generation route");
    assert_eq!(actual, [7; 5]);
    assert_eq!(
        calls, 4,
        "only the transitions between emitted tokens decode"
    );
    let reference = model
        .generate_greedy_uncached(&tokenizer, &rgb, 4, 4, "OCR:", 5)
        .expect("independent full-forward generation");
    assert_eq!(actual, reference);
    assert_eq!(
        model
            .generate_greedy(&tokenizer, &rgb, 4, 4, "OCR:", 5)
            .unwrap(),
        actual,
        "public entry delegates to the observed route"
    );
}

#[test]
fn cached_request_starts_fresh_scratch_for_each_sequence() {
    let model = model(7);
    let tokenizer = tokenizer();
    let rgb = image();
    for budget in [4, 2, 6] {
        let mut calls = 0;
        let output = model
            .generate_greedy_observed(&tokenizer, &rgb, 4, 4, "OCR:", budget, |scratch| {
                calls += 1;
                assert_eq!(scratch.test_decode_calls(), calls);
            })
            .unwrap();
        assert_eq!(output, vec![7; budget]);
        assert_eq!(calls, budget - 1);
    }
}

#[test]
fn zero_budget_skips_prompt_and_scratch() {
    let output = model(7)
        .generate_greedy_observed(&tokenizer(), &[], 0, 0, "", 0, |_| {
            panic!("zero budget cannot decode")
        })
        .unwrap();
    assert!(output.is_empty());
}

#[test]
fn prefill_eos_and_single_token_budget_do_not_decode() {
    let tokenizer = tokenizer();
    let rgb = image();
    for (winner, budget) in [(EOS_ID, 5), (7, 1)] {
        let output = model(winner)
            .generate_greedy_observed(&tokenizer, &rgb, 4, 4, "OCR:", budget, |_| {
                panic!("prefill already satisfied generation")
            })
            .unwrap();
        assert_eq!(output, [winner]);
    }
}

#[test]
fn invalid_image_fails_before_decode_observation() {
    let result = model(7).generate_greedy_observed(&tokenizer(), &[], 4, 4, "OCR:", 3, |_| {
        panic!("invalid image cannot decode")
    });
    assert!(result.is_err());
}

#[test]
fn greedy_token_decisions_preserve_ties_and_stop_order() {
    assert_eq!(argmax_u32(&[0.0, 2.0, 2.0, 1.0]), 1);
    let mut calls = 0;
    let output = greedy_tokens(argmax_u32(&[0.0, 2.0, 2.0]), 5, |token| {
        assert_eq!(token, 1);
        calls += 1;
        Ok(argmax_u32(&[0.0, 1.0, 2.0]))
    })
    .unwrap();
    assert_eq!(output, [1, EOS_ID]);
    assert_eq!(calls, 1);
}
