//! Query-likelihood reranking over the Qwen3.5 causal language-model scoring path.
//!
//! Scores each candidate by `P(query | candidate)`: condition the model on the
//! candidate text and ask how likely it is to continue with the query text. A
//! candidate the model finds highly predictive of the query (low query-token
//! NLL) ranks above one it finds surprising. This is the classic query-likelihood
//! retrieval model, built entirely from [`Qwen35Model::compute_token_nlls`] — no
//! new kernels. Scoring runs through whatever LoRA hook is currently attached to
//! the model (the default no-op hook, or one set via [`Qwen35Model::set_lora`]),
//! since [`Qwen35Model::compute_token_nlls`] drives the same `forward_step` every
//! other entry point uses.

use super::model::Qwen35Model;
use crate::error::InferenceError;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use std::cmp::Ordering;

impl Qwen35Model {
    /// **Unstable**: rerank `candidates` by query-likelihood `P(query | candidate)`.
    ///
    /// For each candidate, tokenizes `[candidate_tokens, query_tokens]` and asks
    /// [`Self::compute_token_nlls`] to score the concatenation. The mean
    /// negative-log-likelihood over the query-token positions (with the
    /// candidate as conditioning context) becomes the candidate's relevance
    /// score, negated so a higher score means a more relevant candidate.
    ///
    /// Returns `(original_index, relevance_score)` pairs sorted by score
    /// descending; ties keep the original candidate order. Deterministic: no
    /// sampling anywhere on this path, and each candidate gets a fresh KV
    /// cache / GDN state (see [`Self::compute_token_nlls`]).
    ///
    /// `candidates` empty returns `Ok(vec![])`. If a candidate's tokens would
    /// push `[candidate, query]` past [`Self::max_context`], the candidate is
    /// left-truncated (earliest tokens dropped, keeping the tokens nearest the
    /// query) so the full query always survives intact. Returns a typed
    /// [`InferenceError::InvalidInput`], never a silent zero score, when the
    /// query alone leaves no room for any candidate context, or when the query
    /// or a (possibly truncated) candidate tokenizes to nothing.
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<(usize, f32)>, InferenceError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // `Tokenizer::tokenize` pads to the tokenizer's configured `max_seq_len`
        // (4096 by default) — or, for input longer than that, truncates while
        // still reporting the pre-truncation `real_length`, which would then
        // index past the truncated `input_ids`. Bumping `max_seq_len` to cover
        // the longest string this call will tokenize keeps every `real_length`
        // accurate. One bump (one vocab clone) up front, reused below, rather
        // than one per candidate.
        let longest = candidates
            .iter()
            .map(|c| c.len())
            .max()
            .unwrap_or(0)
            .max(query.len());
        let tokenizer = self.tokenizer.with_max_seq_len(longest.saturating_add(64));

        let query_tokens = tokenize_unpadded(&tokenizer, query);
        if query_tokens.is_empty() {
            return Err(InferenceError::InvalidInput(
                "rerank: query tokenizes to zero tokens".into(),
            ));
        }

        let max_context = self.max_context();
        if query_tokens.len() >= max_context {
            return Err(InferenceError::InvalidInput(format!(
                "rerank: query ({} tokens) alone leaves no room for candidate \
                 context within the model context window ({max_context})",
                query_tokens.len()
            )));
        }
        let candidate_budget = max_context - query_tokens.len();

        let mut scored = Vec::with_capacity(candidates.len());
        for (idx, candidate) in candidates.iter().enumerate() {
            let mut candidate_tokens = tokenize_unpadded(&tokenizer, candidate);
            if candidate_tokens.len() > candidate_budget {
                // Left-truncate: drop the earliest candidate tokens, keeping the
                // tail (nearest the query) so the full query is never touched.
                let drop = candidate_tokens.len() - candidate_budget;
                candidate_tokens.drain(..drop);
            }
            if candidate_tokens.is_empty() {
                return Err(InferenceError::InvalidInput(format!(
                    "rerank: candidate {idx} tokenizes to zero tokens; query-likelihood \
                     scoring needs at least one candidate token as conditioning context"
                )));
            }

            let candidate_len = candidate_tokens.len();
            let mut sequence = candidate_tokens;
            sequence.extend_from_slice(&query_tokens);

            let nlls = self.compute_token_nlls(&sequence)?;
            // `nlls[i]` scores `sequence[i + 1]`. The query occupies sequence
            // positions `[candidate_len, candidate_len + query_tokens.len())`, so
            // its NLLs live at `nlls[candidate_len - 1 ..= candidate_len + query_tokens.len() - 2]`.
            let query_nlls = &nlls[candidate_len - 1..candidate_len - 1 + query_tokens.len()];
            let mean_nll: f64 =
                query_nlls.iter().map(|&x| x as f64).sum::<f64>() / query_tokens.len() as f64;
            scored.push((idx, -(mean_nll as f32)));
        }

        scored.sort_by(rerank_order);

        Ok(scored)
    }
}

/// Tokenize `text` and return its unpadded token IDs. `tokenizer` must already
/// be bumped to a `max_seq_len` at or above `text`'s real token count (see
/// [`Qwen35Model::rerank`]), or the returned IDs would be silently truncated.
fn tokenize_unpadded(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

/// Sort order for `rerank`'s `(original_index, relevance_score)` pairs:
/// descending by score, ties broken by ascending original index. Mirrors
/// `sampling::candidate_order`'s NaN handling — a NaN score is treated as the
/// worst possible relevance regardless of sign, and the NaN/NaN case falls
/// back to the index tie-break, which keeps this a total order (`sort_by`
/// requires one; comparing two NaNs as merely "equal-ish" is not antisymmetric
/// and can sort nondeterministically with a NaN-heavy input).
fn rerank_order(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::gdn::GatedDeltaNetWeights;
    use crate::lora_hook::NoopLoraHook;
    use crate::model::qwen35::ModelWeights;
    use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
    use crate::rope::RopeTable;

    /// Deterministic xorshift RNG -> uniform noise in `[-scale, scale]`.
    fn rand_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            out.push(((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    /// A tiny byte-level BPE vocabulary big enough to spell short lowercase
    /// English words distinctly (used to build a query whose continuation is
    /// unambiguous from one candidate but not the others).
    fn test_tokenizer() -> BpeTokenizer {
        let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
  "post_processor": null,
  "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, " ": 6, "f": 7, "g": 8,
      "h": 9, "i": 10, "j": 11, "k": 12, "l": 13, "m": 14, "n": 15, "o": 16,
      "p": 17, "q": 18, "r": 19, "s": 20, "t": 21, "u": 22, "v": 23, "w": 24,
      "x": 25, "y": 26, "z": 27
    },
    "merges": []
  }
}"#;
        BpeTokenizer::from_tokenizer_json_str(json).expect("rerank test tokenizer parses")
    }

    fn test_config() -> Qwen35Config {
        let num_hidden_layers = 4;
        let full_attention_interval = 4;
        Qwen35Config {
            hidden_size: 64,
            num_hidden_layers,
            vocab_size: 28,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 4,
            linear_num_value_heads: Some(4),
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval,
            layer_types: compute_layer_types(num_hidden_layers, full_attention_interval),
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: 0,
            max_position_embeddings: 64,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        }
    }

    fn build_model(cfg: Qwen35Config, seed: u64) -> Qwen35Model {
        use crate::model::qwen35::{
            AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
            FullAttentionLayerWeights,
        };

        let mut rng = seed | 1;
        let h = cfg.hidden_size;

        let embed_tokens = rand_vec(&mut rng, cfg.vocab_size * h, 0.02);
        let final_norm = rand_vec(&mut rng, h, 0.02);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_type in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: rand_vec(&mut rng, h, 0.02),
                post_attention_layernorm: rand_vec(&mut rng, h, 0.02),
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    up_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    down_proj: rand_vec(&mut rng, h * cfg.intermediate_size, 0.02),
                }),
            };
            let attn = match layer_type {
                LayerType::LinearAttention => {
                    let qkv_dim = cfg.linear_qkv_dim();
                    let output_dim = cfg.linear_output_dim();
                    let nh = cfg.linear_num_key_heads;
                    let kernel = cfg.linear_conv_kernel_dim;
                    AttentionWeights::Linear(GatedDeltaNetWeights {
                        in_proj_qkv: rand_vec(&mut rng, qkv_dim * h, 0.02),
                        in_proj_qkv_rows: qkv_dim,
                        in_proj_qkv_cols: h,
                        in_proj_z: rand_vec(&mut rng, output_dim * h, 0.02),
                        in_proj_z_rows: output_dim,
                        in_proj_z_cols: h,
                        in_proj_b: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_b_rows: nh,
                        in_proj_b_cols: h,
                        in_proj_a: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_a_rows: nh,
                        in_proj_a_cols: h,
                        a_log: rand_vec(&mut rng, nh, 0.02),
                        dt_bias: rand_vec(&mut rng, nh, 0.02),
                        conv1d_weight: rand_vec(&mut rng, qkv_dim * kernel, 0.02),
                        conv_dim: qkv_dim,
                        kernel_size: kernel,
                        norm_weight: rand_vec(&mut rng, output_dim, 0.02),
                        out_proj: rand_vec(&mut rng, h * output_dim, 0.02),
                        out_proj_rows: h,
                        out_proj_cols: output_dim,
                    })
                }
                LayerType::FullAttention => {
                    let q_dim = cfg.full_q_dim();
                    let kv_dim = cfg.full_kv_dim();
                    AttentionWeights::Full(FullAttentionLayerWeights {
                        q_proj: rand_vec(&mut rng, 2 * q_dim * h, 0.02),
                        k_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        v_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        o_proj: rand_vec(&mut rng, h * q_dim, 0.02),
                        q_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                        k_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                    })
                }
            };
            layers.push((attn, common));
        }

        let rope = RopeTable::new(
            cfg.rope_dim(),
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
        );

        Qwen35Model {
            config: cfg,
            weights: ModelWeights {
                embed_tokens,
                lm_head: None,
                final_norm,
                layers,
            },
            tokenizer: test_tokenizer(),
            rope,
            lora: Box::new(NoopLoraHook),
        }
    }

    #[test]
    fn rerank_empty_candidates_returns_empty() {
        let model = build_model(test_config(), 0xE001);
        let out = model
            .rerank("query text", &[])
            .expect("empty candidates ok");
        assert!(
            out.is_empty(),
            "empty candidate list must return an empty ranking"
        );
    }

    #[test]
    fn rerank_single_candidate_returns_one_result_at_index_zero() {
        let model = build_model(test_config(), 0xE002);
        let out = model
            .rerank("a query", &["a candidate"])
            .expect("single candidate reranks");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert!(out[0].1.is_finite(), "relevance score must be finite");
    }

    #[test]
    fn rerank_is_deterministic() {
        let model = build_model(test_config(), 0xDEC0DE);
        let query = "hello world";
        let candidates = ["good morning", "unrelated noise", "hello there friend"];
        let a = model.rerank(query, &candidates).unwrap();
        let b = model.rerank(query, &candidates).unwrap();
        assert_eq!(
            a, b,
            "rerank must produce bit-identical (index, score) output across calls"
        );
    }

    /// Ranking-correctness: construct a query that is a direct continuation of
    /// one candidate's own vocabulary pattern and unrelated to the others, and
    /// check that candidate ranks first. Uses repeated short words so a tiny
    /// random-weight model still has enough token-identity signal to prefer the
    /// candidate that shares vocabulary with the query over ones that don't.
    ///
    /// Mutation-sensitive by construction: if the query-token slice indexing in
    /// `rerank` is wrong (off-by-one on the NLL bounds, or scoring the whole
    /// sequence instead of just the query positions), the score computed for
    /// each candidate stops being "how well does this candidate predict the
    /// query" and the identity-token/random-model signal this test relies on
    /// breaks — see `rerank_score_matches_direct_query_position_nll_reference`,
    /// which pins the exact contract by recomputing the query-position-only NLL
    /// average independently and asserting equality.
    #[test]
    fn rerank_ranks_matching_candidate_first() {
        let model = build_model(test_config(), 0xA55A);
        let query = "aaa aaa aaa";
        let candidates = ["aaa aaa aaa aaa", "zzz yyy xxx www", "qqq ppp ooo nnn"];
        let ranked = model.rerank(query, &candidates).expect("rerank ok");
        assert_eq!(ranked.len(), 3);
        assert_eq!(
            ranked[0].0, 0,
            "the candidate sharing the query's exact token pattern must rank first; got {ranked:?}"
        );
        // Scores must be strictly descending (no ties in this constructed case).
        assert!(ranked[0].1 > ranked[1].1);
        assert!(ranked[1].1 > ranked[2].1);
    }

    /// Pins the exact score contract by recomputing it independently:
    /// tokenize the same way, call `compute_token_nlls` directly, and average
    /// by hand only the NLLs at the query-token positions. `rerank`'s returned
    /// score must equal this reference exactly (up to float round-off).
    ///
    /// This is the mutation the ranking-correctness test above cannot see: a
    /// mutant that averages the *whole* sequence's NLLs (instead of just the
    /// query positions) can still happen to rank the same candidate first —
    /// found by hand while preparing this suite, see the mutation-sensitivity
    /// note below — but it cannot produce the same numeric score as this
    /// independently-derived reference.
    #[test]
    fn rerank_score_matches_direct_query_position_nll_reference() {
        let model = build_model(test_config(), 0xF00D_BEEF);
        let query = "hello";
        let candidate = "some context words";
        let ranked = model
            .rerank(query, &[candidate])
            .expect("single-candidate rerank ok");

        let tokenizer = model.tokenizer().with_max_seq_len(4096);
        let cand_ids_input = tokenizer.tokenize(candidate);
        let cand_ids = &cand_ids_input.input_ids[..cand_ids_input.real_length];
        let query_ids_input = tokenizer.tokenize(query);
        let query_ids = &query_ids_input.input_ids[..query_ids_input.real_length];

        let mut sequence = cand_ids.to_vec();
        sequence.extend_from_slice(query_ids);
        let nlls = model.compute_token_nlls(&sequence).unwrap();
        let expected_query_nlls = &nlls[cand_ids.len() - 1..cand_ids.len() - 1 + query_ids.len()];
        let expected_mean: f64 =
            expected_query_nlls.iter().map(|&x| x as f64).sum::<f64>() / query_ids.len() as f64;
        let expected_score = -(expected_mean as f32);

        assert!(
            (ranked[0].1 - expected_score).abs() < 1e-6,
            "rerank's score must equal the independently recomputed query-position-only \
             NLL average; got {} expected {expected_score}",
            ranked[0].1
        );
    }

    #[test]
    fn rerank_long_candidate_is_left_truncated_not_rejected() {
        let model = build_model(test_config(), 0xB0B0);
        let max_context = model.max_context();
        // Build a candidate whose token count alone exceeds max_context; rerank
        // must left-truncate it and still return a finite score, not error.
        let long_candidate = "a ".repeat(max_context * 2);
        let out = model
            .rerank("a query", &[long_candidate.as_str()])
            .expect("over-length candidate must be left-truncated, not rejected");
        assert_eq!(out.len(), 1);
        assert!(out[0].1.is_finite());
    }

    #[test]
    fn rerank_query_alone_exceeding_capacity_errors() {
        let model = build_model(test_config(), 0xC0C0);
        let max_context = model.max_context();
        let huge_query = "a ".repeat(max_context * 2);
        let err = model
            .rerank(huge_query.as_str(), &["short candidate"])
            .expect_err(
                "a query alone at/over capacity must error, not panic or truncate silently",
            );
        let msg = format!("{err}");
        assert!(
            msg.contains("context window"),
            "error must name the context window; got: {msg}"
        );
    }

    #[test]
    fn rerank_empty_query_errors() {
        let model = build_model(test_config(), 0xC0FF);
        let err = model
            .rerank("", &["a candidate"])
            .expect_err("an empty query tokenizes to zero tokens and must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("query"),
            "error must mention the query; got: {msg}"
        );
    }

    #[test]
    fn rerank_empty_candidate_string_errors() {
        let model = build_model(test_config(), 0xC0FE);
        let err = model
            .rerank("a query", &["a real candidate", ""])
            .expect_err("an empty-string candidate tokenizes to zero tokens and must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("candidate 1"),
            "error must name the offending candidate index; got: {msg}"
        );
    }

    #[test]
    fn rerank_order_sorts_descending_with_nan_last_and_stable_ties() {
        // NaN != NaN under `==`, so the trailing NaN entry can't go through a
        // whole-vec `assert_eq!` — check the finite prefix by equality and the
        // NaN placement separately.
        let mut scored = [
            (0usize, 1.0f32),
            (1, f32::NAN),
            (2, 3.0),
            (3, 3.0),
            (4, -1.0),
        ];
        scored.sort_by(rerank_order);
        assert_eq!(
            &scored[..4],
            &[(2, 3.0), (3, 3.0), (0, 1.0), (4, -1.0)][..],
            "descending by score; equal scores (2 vs 3) keep ascending index order"
        );
        assert_eq!(scored[4].0, 1, "NaN-scored entry sorts last");
        assert!(scored[4].1.is_nan());
    }

    /// Integration smoke test against the real Qwen3.5-0.8B checkpoint, when
    /// present locally. Self-skips (not `#[ignore]`) so it runs whenever the
    /// checkpoint is available and stays silent in environments without model
    /// weights on disk, mirroring `eval_perplexity`'s `tokenize_with_uncaps_long_corpus`.
    /// Gated on `f16`: the real Qwen3.5-0.8B checkpoint stores `embed_tokens` as
    /// BF16, which the loader rejects without this feature (see the other
    /// `f16`-gated benches/examples in `Cargo.toml`), so without it there is no
    /// safe way to distinguish "checkpoint absent" from "checkpoint present but
    /// unloadable" and this test would spuriously fail on a machine that has
    /// the checkpoint but built without the feature.
    #[cfg(feature = "f16")]
    #[test]
    fn rerank_real_checkpoint_smoke_and_latency() {
        let model_dir =
            std::path::Path::new(concat!(env!("HOME"), "/.lattice/models/qwen3.5-0.8b"));
        if !model_dir.exists() {
            eprintln!(
                "SKIP: no checkpoint at {}; need Qwen3.5-0.8B locally",
                model_dir.display()
            );
            return;
        }
        let model = Qwen35Model::from_safetensors(model_dir).expect("load real checkpoint");

        let query = "What is the boiling point of water at sea level?";
        let candidates = [
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "The stock market closed higher today after a volatile trading session.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "At sea level, water reaches its boiling point at 212 degrees Fahrenheit.",
            "The Great Wall of China being visible from orbit is a popular myth.",
            "A good sourdough bread recipe requires a live starter culture.",
            "Water's boiling point drops at higher altitudes due to lower air pressure.",
            "The mitochondria is the powerhouse of the cell.",
            "Basketball games consist of four quarters of play.",
            "Ice melts into liquid water at zero degrees Celsius.",
        ];

        let start = std::time::Instant::now();
        let ranked = model
            .rerank(query, &candidates)
            .expect("rerank on real checkpoint");
        let elapsed = start.elapsed();

        assert_eq!(ranked.len(), candidates.len());
        for &(idx, score) in &ranked {
            assert!(idx < candidates.len());
            assert!(
                score.is_finite(),
                "score for candidate {idx} must be finite"
            );
        }
        eprintln!(
            "rerank({} candidates) on real Qwen3.5-0.8B checkpoint (CPU): {elapsed:?}; \
             top candidate [{}]: {:?}",
            candidates.len(),
            ranked[0].0,
            candidates[ranked[0].0]
        );
    }
}
