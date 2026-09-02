//! PaddleOCR-VL-1.6 end-to-end CPU forward: image + prompt text ->
//! greedy text.
//!
//! Composes the four independently gated pieces — the image processor
//! (`vision::paddleocr_preprocess`), the SigLIP-shaped vision encoder +
//! projector (`vision::paddleocr_vit`), the ERNIE-4.5 dense decoder
//! (`model::ernie45`), and the ERNIE BPE tokenizer
//! (`tokenizer::gemma_bpe`) — into the reference's own prompt algebra
//! from the checkpoint's `processing_paddleocr_vl` +
//! `modeling_paddleocr_vl` source:
//!
//! 1. **Prompt** — `add_generation_prompt=true` chat template for one
//!    image: `<|begin_of_sentence|>User: <|IMAGE_START|>` + N x
//!    `<|IMAGE_PLACEHOLDER|>` + `<|IMAGE_END|>` + `T\nAssistant:\n`,
//!    tokenized in one pass (`add_bos_token=false`: no extra `<s>`; the
//!    begin-of-sentence token is part of the template text). N is the
//!    post-merge grid: `grid_h / merge * grid_w / merge`.
//! 2. **Splice** — `inputs_embeds = embed_tokens[ids]`, then the rows at
//!    the placeholder positions are replaced, in order, by the projector
//!    output rows.
//! 3. **Positions** — `get_rope_index`: three rows (t, h, w) per token.
//!    Text before the image advances all three by 1 per token; the image
//!    block holds `t = start` while `h`/`w` walk the merged grid in
//!    raster order; text after the image resumes all three from
//!    (max so far) + 1.
//! 4. **Greedy loop** — no KV cache (first slice): every step re-runs the
//!    full forward over prompt + generated so far and takes the argmax of
//!    the last row's logits; generated tokens are text, so their position
//!    columns continue the text-after rule. Stops at eos or
//!    `max_new_tokens`.

use std::path::Path;

use crate::error::InferenceError;
use crate::model::ernie45::{Ernie45Config, Ernie45Model, Ernie45Weights};
use crate::tokenizer::common::Tokenizer;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use crate::vision::paddleocr_preprocess::{
    PaddleOcrImageProcessorConfig, PreprocessedImage, preprocess_rgb8,
};
use crate::vision::paddleocr_vit::{
    PaddleOcrVisionConfig, PaddleOcrVisionWeights, paddleocr_vision_forward,
};
use crate::weights::SafetensorsFile;

/// The checkpoint's special ids (`modeling_paddleocr_vl.py`): the vision
/// tokens the processor writes into the template and the stop token.
pub const IMAGE_PLACEHOLDER_ID: u32 = 100_295;
pub const IMAGE_START_ID: u32 = 101_305;
pub const IMAGE_END_ID: u32 = 101_306;
pub const EOS_ID: u32 = 2;

/// One image's prefill result — everything the golden gate compares.
pub struct PrefillTrace {
    /// `add_generation_prompt=true` template ids (exact, unpadded).
    pub prompt_ids: Vec<u32>,
    /// `get_rope_index` output: one `[t, h, w]` per prompt token.
    pub positions: Vec<[u32; 3]>,
    /// Projector output rows `[n_image_tokens, text_hidden]` in merged
    /// raster order — the rows spliced into the placeholder positions.
    pub projector_rows: Vec<f32>,
    /// `embed_tokens[prompt_ids]` with the placeholder rows replaced by
    /// `projector_rows`, in order.
    pub spliced_embeds: Vec<f32>,
    /// Full-sequence logits `[seq_len, vocab]`.
    pub logits: Vec<f32>,
    /// `(temporal, grid_h, grid_w)` from the image processor.
    pub grid_thw: (usize, usize, usize),
    /// `(height, width)` after smart-resize.
    pub resized_hw: (usize, usize),
}

/// Image + text -> greedy tokens, over the full PaddleOCR-VL-1.6
/// checkpoint (decoder + vision + image processor). The tokenizer is
/// passed in per call: it is a text-only asset, kept out of the weight
/// load so a checkpoint directory without a `tokenizer.json` (as on some
/// deployment targets) still loads for inference with the tokenizer
/// supplied separately.
pub struct PaddleOcrVlModel {
    decoder: Ernie45Model,
    decoder_cfg: Ernie45Config,
    vision_cfg: PaddleOcrVisionConfig,
    vision_weights: PaddleOcrVisionWeights,
    processor_cfg: PaddleOcrImageProcessorConfig,
}

impl PaddleOcrVlModel {
    /// Load the whole checkpoint from `model_dir` (single-shard
    /// `model.safetensors`, `config.json`, `preprocessor_config.json`).
    ///
    /// # Errors
    ///
    /// Any loader error: missing file, unparseable JSON, shape mismatch,
    /// or a config this reference does not implement.
    pub fn load(model_dir: &Path) -> Result<Self, InferenceError> {
        let decoder_cfg = Ernie45Config::from_config_json(&model_dir.join("config.json"))?;
        let vision_cfg = PaddleOcrVisionConfig::from_config_json(&model_dir.join("config.json"))?;
        if vision_cfg.text_hidden_size != decoder_cfg.hidden_size {
            return Err(InferenceError::Inference(format!(
                "vision projector targets text_hidden {} but the decoder hidden is {}",
                vision_cfg.text_hidden_size, decoder_cfg.hidden_size
            )));
        }
        let processor_cfg = PaddleOcrImageProcessorConfig::from_preprocessor_json(
            &model_dir.join("preprocessor_config.json"),
        )?;
        let mut source = SafetensorsFile::open(&model_dir.join("model.safetensors"))?;
        let decoder_weights = Ernie45Weights::load(&mut source, &decoder_cfg)?;
        let vision_weights = PaddleOcrVisionWeights::load(&mut source, &vision_cfg)?;
        Ok(Self {
            decoder: Ernie45Model::new(decoder_cfg.clone(), decoder_weights)?,
            decoder_cfg,
            vision_cfg,
            vision_weights,
            processor_cfg,
        })
    }

    /// `add_generation_prompt=true` chat template for one image, tokenized
    /// in one pass exactly as the reference processor does (special tokens
    /// matched as single ids, no BOS prepended).
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] if the grid is not a multiple of the
    /// merge kernel (so N is not an integer) or the token stream does not
    /// contain exactly N placeholders.
    pub fn build_prompt_ids(
        &self,
        tokenizer: &GemmaBpeTokenizer,
        grid_thw: (usize, usize, usize),
        text: &str,
    ) -> Result<Vec<u32>, InferenceError> {
        let m = self.vision_cfg.spatial_merge_size;
        let (_, grid_h, grid_w) = grid_thw;
        if grid_h % m != 0 || grid_w % m != 0 {
            return Err(InferenceError::Inference(format!(
                "grid {grid_h}x{grid_w} is not a multiple of the {m}x{m} merge kernel"
            )));
        }
        let n = (grid_h / m) * (grid_w / m);
        let placeholders = "<|IMAGE_PLACEHOLDER|>".repeat(n);
        let template = format!(
            "<|begin_of_sentence|>User: <|IMAGE_START|>{placeholders}<|IMAGE_END|>{text}\nAssistant:\n"
        );
        let tokenized = tokenizer.tokenize(&template);
        let ids = tokenized.input_ids[..tokenized.real_length].to_vec();
        let count = ids.iter().filter(|&&id| id == IMAGE_PLACEHOLDER_ID).count();
        if count != n {
            return Err(InferenceError::Inference(format!(
                "template tokenized with {count} image placeholders; the grid {grid_thw:?} requires {n}"
            )));
        }
        Ok(ids)
    }

    /// `get_rope_index` for one image: three position rows per token.
    ///
    /// - Text before the first placeholder: `[i, i, i]` (all rows advance
    ///   together).
    /// - The image block (N contiguous placeholders, merged-grid raster
    ///   order `gh x gw`): `[st, st + row, st + col]` with `st` the text
    ///   length so far.
    /// - Text after the image: `[p, p, p]` continuing from (max so far) + 1
    ///   per token.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] when the placeholder count does not
    /// match `grid_h/merge * grid_w/merge` or the placeholders are not
    /// contiguous (the reference template always emits one block).
    pub fn rope_index(
        ids: &[u32],
        grid_thw: (usize, usize, usize),
        merge: usize,
    ) -> Result<Vec<[u32; 3]>, InferenceError> {
        if merge == 0 {
            return Err(InferenceError::Inference("merge kernel is zero".into()));
        }
        let (t, grid_h, grid_w) = grid_thw;
        if t != 1 {
            return Err(InferenceError::Inference(format!(
                "grid_thw temporal {t} != 1: this reference is image-only"
            )));
        }
        if grid_h % merge != 0 || grid_w % merge != 0 {
            return Err(InferenceError::Inference(format!(
                "grid {grid_h}x{grid_w} is not a multiple of the {merge}x{merge} merge kernel"
            )));
        }
        let gh = grid_h / merge;
        let gw = grid_w / merge;
        let n = gh * gw;

        let placeholder_idx: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == IMAGE_PLACEHOLDER_ID)
            .map(|(i, _)| i)
            .collect();
        if placeholder_idx.len() != n {
            return Err(InferenceError::Inference(format!(
                "token stream has {} image placeholders; the grid {grid_thw:?} requires {n}",
                placeholder_idx.len()
            )));
        }
        let start = placeholder_idx[0];
        for (offset, &idx) in placeholder_idx.iter().enumerate() {
            if idx != start + offset {
                return Err(InferenceError::Inference(format!(
                    "image placeholders are not contiguous (first at {start}, offset {offset} at {idx})"
                )));
            }
        }

        let mut out: Vec<[u32; 3]> = Vec::with_capacity(ids.len());
        // Text before the image.
        for i in 0..start {
            out.push([i as u32, i as u32, i as u32]);
        }
        // The image block.
        for p in 0..n {
            let (row, col) = ((p / gw) as u32, (p % gw) as u32);
            out.push([start as u32, start as u32 + row, start as u32 + col]);
        }
        // Text after the image: max over all previous position values + 1,
        // then +1 per token. At the block end the max is
        // start + max(gh - 1, gw - 1); after that the three rows stay
        // equal, so a single counter suffices.
        let mut p = (start + (gh - 1).max(gw - 1)) as u32;
        for _ in start + n..ids.len() {
            p += 1;
            out.push([p, p, p]);
        }
        Ok(out)
    }

    /// Preprocess the image, run the vision encoder + projector, assemble
    /// the prompt, splice, and run the full-sequence decoder forward.
    ///
    /// `rgb` is interleaved HWC RGB8, length `h * w * 3`.
    ///
    /// # Errors
    ///
    /// Any pipeline error, or a placeholder/projector row count mismatch
    /// (fail-closed rather than silently splicing the wrong rows).
    pub fn prefill(
        &self,
        tokenizer: &GemmaBpeTokenizer,
        rgb: &[u8],
        h: usize,
        w: usize,
        text: &str,
    ) -> Result<PrefillTrace, InferenceError> {
        let PreprocessedImage {
            pixel_values,
            grid_thw,
            resized_hw,
        } = preprocess_rgb8(&self.processor_cfg, rgb, h, w)?;

        let projector = paddleocr_vision_forward(
            &self.vision_weights,
            &self.vision_cfg,
            &pixel_values,
            grid_thw.1,
            grid_thw.2,
        )?;

        let prompt_ids = self.build_prompt_ids(tokenizer, grid_thw, text)?;
        let hidden = self.decoder_cfg.hidden_size;
        let n_img = (grid_thw.1 / self.vision_cfg.spatial_merge_size)
            * (grid_thw.2 / self.vision_cfg.spatial_merge_size);
        if projector.len() != n_img * hidden {
            return Err(InferenceError::Inference(format!(
                "projector produced {} values; expected {n_img} rows x {hidden}",
                projector.len()
            )));
        }

        let embeds = &self.decoder.embed_tokens();
        let mut spliced = vec![0f32; prompt_ids.len() * hidden];
        let mut k = 0usize;
        for (pos, &id) in prompt_ids.iter().enumerate() {
            let row = &mut spliced[pos * hidden..(pos + 1) * hidden];
            if id == IMAGE_PLACEHOLDER_ID {
                row.copy_from_slice(&projector[k * hidden..(k + 1) * hidden]);
                k += 1;
            } else {
                row.copy_from_slice(&embeds[id as usize * hidden..][..hidden]);
            }
        }
        if k != n_img {
            return Err(InferenceError::Inference(format!(
                "spliced {k} projector rows into placeholders; the grid {grid_thw:?} has {n_img}"
            )));
        }

        let positions =
            Self::rope_index(&prompt_ids, grid_thw, self.vision_cfg.spatial_merge_size)?;
        let trace = self.decoder.forward_embeds_trace(&spliced, &positions)?;

        Ok(PrefillTrace {
            prompt_ids,
            positions,
            projector_rows: projector,
            spliced_embeds: spliced,
            logits: trace.logits,
            grid_thw,
            resized_hw,
        })
    }

    /// Greedy decode over `max_new_tokens` steps with **no KV cache**:
    /// every step re-runs the full forward over the whole sequence (first
    /// slice; a cached decode is a follow-up). `rgb` is interleaved HWC
    /// RGB8, length `h * w * 3`.
    ///
    /// # Errors
    ///
    /// Any pipeline error; stops early at `EOS_ID` or when
    /// `max_new_tokens` tokens have been produced.
    pub fn generate_greedy(
        &self,
        tokenizer: &GemmaBpeTokenizer,
        rgb: &[u8],
        h: usize,
        w: usize,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, InferenceError> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let hidden = self.decoder_cfg.hidden_size;
        let vocab = self.decoder_cfg.vocab_size;
        let prefill = self.prefill(tokenizer, rgb, h, w, text)?;

        let mut ids = prefill.prompt_ids;
        let mut embeds = prefill.spliced_embeds;
        let mut positions = prefill.positions;
        let mut logits = prefill.logits;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let last = &logits[(ids.len() - 1) * vocab..][..vocab];
            let next = argmax_u32(last);
            generated.push(next);
            if next == EOS_ID {
                break;
            }
            // Extend the sequence by one text token and re-forward.
            ids.push(next);
            let embed_len = embeds.len();
            embeds.resize(embed_len + hidden, 0.0);
            embeds[embed_len..]
                .copy_from_slice(&self.decoder.embed_tokens()[next as usize * hidden..][..hidden]);
            let p = positions
                .iter()
                .fold(0u32, |m, r| m.max(r[0]).max(r[1]).max(r[2]))
                + 1;
            positions.push([p, p, p]);
            logits = self
                .decoder
                .forward_embeds_trace(&embeds, &positions)?
                .logits;
        }
        Ok(generated)
    }

    pub fn config(&self) -> &Ernie45Config {
        &self.decoder_cfg
    }

    pub fn processor_config(&self) -> &PaddleOcrImageProcessorConfig {
        &self.processor_cfg
    }
}

/// Plain argmax; ties break to the lowest id (the reference's
/// `torch.argmax` behavior).
pub fn argmax_u32(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ernie45::Ernie45LayerWeights;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/paddleocr_vl")
    }

    fn e2e_golden() -> serde_json::Value {
        let path = fixture_dir().join("e2e/e2e_goldens.json");
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read e2e goldens"))
            .expect("valid e2e_goldens.json")
    }

    fn tokenizer() -> GemmaBpeTokenizer {
        let path = fixture_dir().join("tokenizer/tokenizer.json");
        GemmaBpeTokenizer::from_ernie_tokenizer_json(&path).expect("pinned tokenizer loads")
    }

    const MERGE: usize = 2;

    /// A minimal model standing in for `load` (no checkpoint needed): only
    /// the tokenizer and merge kernel are exercised by these tests.
    fn shell_model() -> PaddleOcrVlModel {
        PaddleOcrVlModel {
            decoder: empty_decoder(),
            decoder_cfg: empty_decoder().config().clone(),
            vision_cfg: PaddleOcrVisionConfig {
                hidden_size: 1152,
                intermediate_size: 4304,
                num_hidden_layers: 27,
                num_attention_heads: 16,
                num_channels: 3,
                patch_size: 14,
                image_size: 384,
                layer_norm_eps: 1e-6,
                spatial_merge_size: MERGE,
                text_hidden_size: 1024,
            },
            vision_weights: PaddleOcrVisionWeights {
                patch_weight: Vec::new(),
                patch_bias: Vec::new(),
                pos_embed: Vec::new(),
                layers: Vec::new(),
                post_ln_weight: Vec::new(),
                post_ln_bias: Vec::new(),
                proj_norm_weight: Vec::new(),
                proj_norm_bias: Vec::new(),
                proj_l1_weight: Vec::new(),
                proj_l1_bias: Vec::new(),
                proj_l2_weight: Vec::new(),
                proj_l2_bias: Vec::new(),
            },
            processor_cfg: PaddleOcrImageProcessorConfig::paddleocr_vl_defaults(),
        }
    }

    fn empty_decoder() -> Ernie45Model {
        let cfg = Ernie45Config::from_config_json_str(
            r#"{"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
                "vocab_size": 300, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 2, 2, 2]},
                "tie_word_embeddings": false, "use_bias": false}"#,
        )
        .expect("test config");
        Ernie45Model::new(
            cfg,
            Ernie45Weights {
                embed_tokens: vec![0f32; 300 * 128],
                layers: vec![
                    Ernie45LayerWeights {
                        q_proj: vec![0f32; 128 * 128],
                        k_proj: vec![0f32; 32 * 128],
                        v_proj: vec![0f32; 32 * 128],
                        o_proj: vec![0f32; 128 * 128],
                        gate_proj: vec![0f32; 256 * 128],
                        up_proj: vec![0f32; 256 * 128],
                        down_proj: vec![0f32; 128 * 256],
                        input_layernorm: vec![1f32; 128],
                        post_attention_layernorm: vec![1f32; 128],
                    },
                    Ernie45LayerWeights {
                        q_proj: vec![0f32; 128 * 128],
                        k_proj: vec![0f32; 32 * 128],
                        v_proj: vec![0f32; 32 * 128],
                        o_proj: vec![0f32; 128 * 128],
                        gate_proj: vec![0f32; 256 * 128],
                        up_proj: vec![0f32; 256 * 128],
                        down_proj: vec![0f32; 128 * 256],
                        input_layernorm: vec![1f32; 128],
                        post_attention_layernorm: vec![1f32; 128],
                    },
                ],
                final_norm: vec![1f32; 128],
                lm_head: vec![0f32; 300 * 128],
            },
        )
        .expect("test decoder")
    }

    #[test]
    fn prompt_assembly_matches_hf_input_ids() {
        let golden = e2e_golden();
        let grid_thw: (usize, usize, usize) = (
            golden["image"]["grid_thw"][0].as_u64().unwrap() as usize,
            golden["image"]["grid_thw"][1].as_u64().unwrap() as usize,
            golden["image"]["grid_thw"][2].as_u64().unwrap() as usize,
        );
        let text = golden["prompt_text"].as_str().unwrap();
        let ids = shell_model()
            .build_prompt_ids(&tokenizer(), grid_thw, text)
            .expect("prompt builds");
        let expect: Vec<u32> = golden["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            ids, expect,
            "prompt ids diverge from the HF one-pass tokenization"
        );
    }

    /// The fixture's own ids re-derived through `rope_index` must equal the
    /// golden `position_ids` — the strongest no-checkpoint check that the
    /// text-before / image-block / text-after rules are correct.
    #[test]
    fn rope_index_matches_hf_golden_positions() {
        let golden = e2e_golden();
        let ids: Vec<u32> = golden["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let grid_thw: (usize, usize, usize) = (
            golden["image"]["grid_thw"][0].as_u64().unwrap() as usize,
            golden["image"]["grid_thw"][1].as_u64().unwrap() as usize,
            golden["image"]["grid_thw"][2].as_u64().unwrap() as usize,
        );
        let got = PaddleOcrVlModel::rope_index(&ids, grid_thw, MERGE).expect("rope index");
        for (row_idx, row_name) in [(0usize, "t"), (1, "h"), (2, "w")] {
            for (c, (g, e)) in got
                .iter()
                .map(|p| p[row_idx])
                .zip(golden["position_ids"][row_idx].as_array().unwrap().iter())
                .enumerate()
            {
                assert_eq!(g, e.as_u64().unwrap() as u32, "row {row_name} col {c}");
            }
        }
        assert_eq!(
            golden["rope_delta"].as_i64().unwrap(),
            (got.iter().map(|p| p[0].max(p[1]).max(p[2])).max().unwrap() as i64 + 1) as i64
                - ids.len() as i64,
            "rope_delta bookkeeping"
        );
    }

    #[test]
    fn rope_index_hand_computed_small_cases() {
        // 18x32 grid, merge 2 -> 9x16 merged grid, 144 placeholders.
        let grid = (1, 18, 32);
        let ids: Vec<u32> = [
            // 5 text tokens before the image.
            1, 2, 3, 4, 5,
        ]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE_PLACEHOLDER_ID, 144))
        .chain([6, 7]) // 2 text tokens after the image.
        .collect();
        let pos = PaddleOcrVlModel::rope_index(&ids, grid, MERGE).expect("rope index");
        assert_eq!(pos.len(), 151);
        // Text before.
        assert_eq!(pos[0], [0, 0, 0]);
        assert_eq!(pos[4], [4, 4, 4]);
        // First placeholder: st = 5.
        assert_eq!(pos[5], [5, 5, 5]);
        // Raster: (row, col) = (p / 16, p % 16).
        assert_eq!(pos[5 + 16], [5, 6, 5], "row 1 col 0");
        assert_eq!(pos[5 + 16 + 7], [5, 6, 12], "row 1 col 7");
        assert_eq!(
            pos[5 + 143],
            [5, 13, 20],
            "last placeholder (row 8, col 15)"
        );
        // Text after: max so far = 5 + max(8, 15) = 20, then +1.
        assert_eq!(pos[149], [21, 21, 21]);
        assert_eq!(pos[150], [22, 22, 22]);
    }

    #[test]
    fn rope_index_rejects_bad_placeholder_shapes() {
        let grid = (1, 18, 32);
        // Too few placeholders.
        let ids: Vec<u32> = [1, 2]
            .into_iter()
            .chain(std::iter::repeat_n(IMAGE_PLACEHOLDER_ID, 143))
            .collect();
        assert!(PaddleOcrVlModel::rope_index(&ids, grid, MERGE).is_err());
        // Non-contiguous placeholders (one moved after the block).
        let mut ids: Vec<u32> = [1, 2]
            .into_iter()
            .chain(std::iter::repeat_n(IMAGE_PLACEHOLDER_ID, 143))
            .collect();
        ids.push(99);
        ids.push(IMAGE_PLACEHOLDER_ID);
        assert!(PaddleOcrVlModel::rope_index(&ids, grid, MERGE).is_err());
        // No placeholders at all.
        assert!(PaddleOcrVlModel::rope_index(&[1, 2, 3], grid, MERGE).is_err());
        // Temporal > 1.
        let ids: Vec<u32> = [1, 2]
            .into_iter()
            .chain(std::iter::repeat_n(IMAGE_PLACEHOLDER_ID, 144))
            .collect();
        assert!(PaddleOcrVlModel::rope_index(&ids, (2, 18, 32), MERGE).is_err());
    }

    #[test]
    fn build_prompt_ids_rejects_grid_not_a_multiple_of_merge() {
        assert!(
            shell_model()
                .build_prompt_ids(&tokenizer(), (1, 18, 33), "OCR:")
                .is_err()
        );
    }

    #[test]
    fn argmax_breaks_ties_to_lowest_id() {
        assert_eq!(argmax_u32(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_u32(&[-5.0, -2.0, -2.0]), 1);
    }

    /// The splicing arithmetic against a synthetic projector (no
    /// checkpoint): rows land exactly at the placeholder positions, in
    /// order, and text rows come from the embedding table.
    #[test]
    fn splice_layout_matches_placeholder_positions() {
        let hidden = 4usize;
        let ids: Vec<u32> = vec![
            10,
            IMAGE_PLACEHOLDER_ID,
            11,
            IMAGE_PLACEHOLDER_ID,
            IMAGE_PLACEHOLDER_ID,
            12,
        ];
        let projector: Vec<f32> = (0..(3 * hidden)).map(|v| v as f32).collect();
        let embeds: Vec<f32> = (0..(13 * hidden)).map(|v| (v + 100) as f32).collect();
        let mut spliced = vec![0f32; ids.len() * hidden];
        let mut k = 0;
        for (pos, &id) in ids.iter().enumerate() {
            let row = &mut spliced[pos * hidden..(pos + 1) * hidden];
            if id == IMAGE_PLACEHOLDER_ID {
                row.copy_from_slice(&projector[k * hidden..(k + 1) * hidden]);
                k += 1;
            } else {
                row.copy_from_slice(&embeds[id as usize * hidden..][..hidden]);
            }
        }
        assert_eq!(k, 3);
        for (pos, expect_src) in [
            (0usize, 10),
            (1, usize::MAX),
            (2, 11),
            (3, usize::MAX),
            (4, usize::MAX),
            (5, 12),
        ] {
            let row = &spliced[pos * hidden..(pos + 1) * hidden];
            if expect_src == usize::MAX {
                // projector row order: rows 0,1,2 -> positions 1,3,4
                let k = match pos {
                    1 => 0,
                    3 => 1,
                    _ => 2,
                };
                assert_eq!(*row, projector[k * hidden..(k + 1) * hidden], "row {pos}");
            } else {
                assert_eq!(
                    *row,
                    embeds[expect_src * hidden..(expect_src + 1) * hidden],
                    "row {pos}"
                );
            }
        }
    }
}
