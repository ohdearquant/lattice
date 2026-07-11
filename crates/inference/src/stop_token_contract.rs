//! Cross-path stop-token contract sweep (#613).
//!
//! Before this fix, the crate had two incompatible contracts for what happens
//! to the token that triggers a stop condition (`eos_token_id` or
//! `stop_token_ids`):
//!
//! - **INCLUDE**: the stop token is pushed into `token_ids`/`text` and then
//!   generation halts (the generic [`crate::generate::generate`] path, and the
//!   Metal MTP/self-speculative decode loops in
//!   [`crate::forward::metal_qwen35`]).
//! - **EXCLUDE**: the stop token is detected *before* it is pushed, so
//!   `token_ids`/`text` never contain it (the Qwen3.5 CPU family in
//!   [`crate::model::qwen35::generation`], [`crate::forward::cpu_f16`],
//!   [`crate::forward::cpu_q8`], [`crate::forward::neon_forward`],
//!   [`crate::forward::batch_prefill`], and the Metal plain/streaming/
//!   prefix-cache paths in [`crate::forward::metal_qwen35`]).
//!
//! **Chosen contract: EXCLUDE.** It was already the majority behaviour (8 of
//! 10 surveyed call sites), it matches the OpenAI/HF `stop` semantics most
//! callers expect (the stop marker is a control signal, not content), and it
//! composes correctly with string-level `stop_strings` truncation, which by
//! construction already drops the matched suffix. `GenerateOutput`'s doc
//! comment (in [`crate::generate`] and [`crate::model::qwen35_config`]) now
//! records this explicitly. The MTP and self-spec loops were the two sites
//! that needed a production change; see `crates/inference/src/forward/metal_qwen35.rs`.
//!
//! ## Entry-point manifest
//!
//! | # | Entry point | Coverage |
//! |---|---|---|
//! | 1 | `Qwen35Model::generate` (f32) | this file |
//! | 2 | `Qwen35Model::generate_streaming` (f32) | this file |
//! | 3 | `Qwen35Model::generate_with_batch_prefill` | this file |
//! | 4 | `forward::cpu_f16::generate_f16` | this file |
//! | 5 | `forward::cpu_q8::generate_q8` | this file |
//! | 6 | `forward::neon_forward::generate_q8_neon` | this file |
//! | 7 | Metal `generate` (plain, no MTP/self-spec) | `forward::metal_qwen35::inner::tests` |
//! | 8 | Metal `generate` with `LATTICE_MTP=1` | `forward::metal_qwen35::inner::tests` + `mtp_greedy_round_tests` |
//! | 9 | Metal `generate` with `LATTICE_SELF_SPEC=1` | `forward::metal_qwen35::inner::tests` + `self_spec_eos_tests` |
//! | 10 | Metal `generate_streaming` | `forward::metal_qwen35::inner::tests` |
//! | 11 | Metal `generate_streaming_with_prefix_cache` | `forward::metal_qwen35::inner::tests` |
//! | 12 | Metal `generate_multimodal` | `forward::metal_qwen35::inner::tests` |
//!
//! Entry 12 (`generate_multimodal`) has its own independent sampling loop —
//! it does not call `generate`/`generate_streaming` internally — and was
//! missing from this manifest until the PR #632 review flagged it as
//! an unguarded sibling-invocation path (#613's own warning pattern). Its
//! loop already used the EXCLUDE contract (checks `is_stop` before pushing
//! at every termination point: first-token sample, KV-full break, and the
//! autoregressive decode loop), so no production change was needed here —
//! only manifest + test coverage.
//!
//! The Metal-family tests (7-11) live inside `metal_qwen35.rs`'s existing
//! `mod tests` rather than here because its GPU test fixtures
//! (`tiny_hybrid_fixture`, `minimal_bpe_tokenizer`, `with_self_spec_env`) are
//! private to that module and already proven correct by dozens of existing
//! tests; duplicating or relocating them would add risk (see the module's
//! `MetalQwen35State` construction path, which needs a live `MTLDevice`) for
//! no contract-coverage benefit. Grep `stop_token_contract` or `#613` in
//! `metal_qwen35.rs` to find them.
//!
//! **`crate::generate::generate` (the generic `QwenModel` path) is
//! intentionally excluded from live coverage.** `QwenModel` holds
//! `ManuallyDrop<QwenWeights<'static>>` fields behind a hand-written unsafe
//! `Drop` impl that enforces a specific field drop order; building a
//! throwaway test instance safely requires reproducing that unsafe
//! self-referential setup outside of the real loader, which is
//! disproportionate risk for a path whose stop-check fix is two one-line
//! reorderings (already applied, see `crates/inference/src/generate.rs`).
//! That fix is covered by reading the diff and the existing
//! `crate::generate` unit tests that exercise `should_stop_token`, not by a
//! new end-to-end fixture here.
//!
//! ## What "sweep" means here
//!
//! Every test below builds the smallest possible zero-layer model (no
//! attention/FFN, just `embed -> final_norm -> lm_head`) with all-zero
//! weights, so greedy sampling deterministically always picks token 0.
//! `eos_token_id` is set to a different value (5) and `stop_token_ids =
//! [0]`, so hitting the stop condition on the very first generated token is
//! unambiguous and reproducible. Each test then asserts the EXCLUDE contract:
//! `generated_tokens == 0`, `token_ids` and `text` empty, `stopped == true`,
//! `stop_reason == Some(StopReason::Eos)`.
//!
//! Most of the paths tested here (1-6) were already EXCLUDE before this
//! session's fix — they regression-lock behaviour that was already correct.
//! The two paths whose logic changed (MTP, self-spec) are unit-tested at the
//! decision-function level in `metal_qwen35.rs`'s `mtp_greedy_round_tests`
//! and `self_spec_eos_tests`, where the same all-zero-weights style
//! before/after vectors were hand-verified to flip from INCLUDE to EXCLUDE
//! when the production fix is reverted.

#[cfg(test)]
mod tests {
    use crate::forward::cpu_f16::generate_f16;
    use crate::forward::cpu_q8::generate_q8;
    use crate::forward::neon::pack_weights_q8;
    use crate::forward::neon_forward::{Q8NeonModel, generate_q8_neon};
    use crate::lora_hook::NoopLoraHook;
    use crate::model::qwen35::{ModelWeights, Qwen35Model};
    use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
    use crate::rope::RopeTable;
    use crate::stop_reason::StopReason;
    use crate::tokenizer::bpe::BpeTokenizer;
    use crate::weights::f16_weights::F16ModelWeights;
    use crate::weights::q8_weights::Q8ModelWeights;
    use std::collections::HashMap;

    const HIDDEN: usize = 4;
    const VOCAB: usize = 8;
    const EOS: u32 = 5;
    const STOP: u32 = 0; // greedy always picks 0 under all-zero weights

    /// Shared zero-layer config: embed -> final_norm -> lm_head only, no
    /// attention/FFN. Mirrors the established fixture recipe already used by
    /// `cpu_f16::zero_layer_f16_fixture`, `cpu_q8::zero_layer_q8_fixture`, and
    /// `batch_prefill::zero_layer_batch_prefill_fixture`.
    fn zero_layer_config() -> Qwen35Config {
        Qwen35Config {
            hidden_size: HIDDEN,
            num_hidden_layers: 0,
            vocab_size: VOCAB,
            intermediate_size: 4,
            rms_norm_eps: 1e-6,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            rope_parameters: None,
            linear_num_key_heads: 1,
            linear_num_value_heads: Some(1),
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 2,
            layer_types: vec![],
            layer_mask: vec![],
            // eos is 5 so that greedy token 0 is NOT eos — this makes
            // stop_token_ids=[0] a distinct, detectable stop signal.
            eos_token_id: EOS,
            max_position_embeddings: 512,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
        }
    }

    /// rope_dim = head_dim * partial_rotary_factor = 4 * 0.5 = 2.
    fn minimal_rope() -> RopeTable {
        RopeTable::new(2, 64, 10_000.0)
    }

    fn minimal_tokenizer() -> BpeTokenizer {
        let mut vocab_map: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
            vocab_map.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap()
    }

    fn stop_gen_cfg() -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: 4,
            stop_token_ids: vec![STOP],
            temperature: 0.0, // greedy: all-zero logits always yield token 0
            ..Default::default()
        }
    }

    /// The #613 contract: a stop token is EXCLUDED from `token_ids`/`text`.
    /// `stopped` is true and `stop_reason` is `Eos` because budget (4 tokens)
    /// was available when the stop fired — a genuine stop, not a length cap.
    fn assert_excludes_stop_token(out: &GenerateOutput, entry_point: &str) {
        assert_eq!(
            out.generated_tokens, 0,
            "{entry_point}: stop-token contract violated — generated_tokens \
             should be 0 (token excluded), got {}",
            out.generated_tokens
        );
        assert!(
            out.token_ids.is_empty(),
            "{entry_point}: stop-token contract violated — token_ids should be \
             empty, got {:?}",
            out.token_ids
        );
        assert!(
            out.text.is_empty(),
            "{entry_point}: stop-token contract violated — text should be \
             empty, got {:?}",
            out.text
        );
        assert!(
            out.stopped,
            "{entry_point}: stopped must be true when a stop token is hit with \
             budget available"
        );
        assert_eq!(
            out.stop_reason,
            Some(StopReason::Eos),
            "{entry_point}: stop_reason must be Eos"
        );
    }

    fn zero_layer_qwen35_model() -> Qwen35Model {
        Qwen35Model {
            config: zero_layer_config(),
            weights: ModelWeights {
                embed_tokens: vec![0.0f32; VOCAB * HIDDEN],
                lm_head: None,
                final_norm: vec![0.0f32; HIDDEN],
                layers: vec![],
            },
            tokenizer: minimal_tokenizer(),
            rope: minimal_rope(),
            lora: Box::new(NoopLoraHook),
        }
    }

    #[test]
    fn qwen35_model_generate_excludes_stop_token() {
        let model = zero_layer_qwen35_model();
        let out = model
            .generate("h", &stop_gen_cfg())
            .expect("generate must succeed");
        assert_excludes_stop_token(&out, "Qwen35Model::generate");
    }

    #[test]
    fn qwen35_model_generate_streaming_excludes_stop_token() {
        let model = zero_layer_qwen35_model();
        let mut deltas: Vec<String> = vec![];
        let out = model
            .generate_streaming("h", &stop_gen_cfg(), |s| deltas.push(s.to_string()))
            .expect("generate_streaming must succeed");
        assert_excludes_stop_token(&out, "Qwen35Model::generate_streaming");
        assert!(
            deltas.is_empty(),
            "generate_streaming must not emit an on_token callback for an \
             excluded stop token, got {deltas:?}"
        );
    }

    #[test]
    #[allow(deprecated)] // exercises the deprecated path itself during its deprecation window (issue #807)
    fn qwen35_model_generate_with_batch_prefill_excludes_stop_token() {
        let model = zero_layer_qwen35_model();
        let out = model
            .generate_with_batch_prefill("h", &stop_gen_cfg())
            .expect("generate_with_batch_prefill must succeed");
        assert_excludes_stop_token(&out, "Qwen35Model::generate_with_batch_prefill");
    }

    #[test]
    fn generate_f16_excludes_stop_token() {
        let cfg = zero_layer_config();
        let weights = F16ModelWeights {
            embed_tokens: vec![0u16; VOCAB * HIDDEN],
            final_norm: vec![0.0f32; HIDDEN],
            layers: vec![],
        };
        let out = generate_f16(
            &weights,
            &cfg,
            &minimal_tokenizer(),
            &minimal_rope(),
            "h",
            &stop_gen_cfg(),
        )
        .expect("generate_f16 must succeed");
        assert_excludes_stop_token(&out, "generate_f16");
    }

    #[test]
    fn generate_q8_excludes_stop_token() {
        let cfg = zero_layer_config();
        let weights = Q8ModelWeights {
            embed_tokens: vec![0.0f32; VOCAB * HIDDEN],
            final_norm: vec![0.0f32; HIDDEN],
            layers: vec![],
        };
        let out = generate_q8(
            &weights,
            &cfg,
            &minimal_tokenizer(),
            &minimal_rope(),
            "h",
            &stop_gen_cfg(),
        )
        .expect("generate_q8 must succeed");
        assert_excludes_stop_token(&out, "generate_q8");
    }

    #[test]
    fn generate_q8_neon_excludes_stop_token() {
        // pack_weights_q8 requires hidden % QK8_0(32) == 0, unlike the other
        // fixtures' HIDDEN=4 — use a dedicated hidden dim here. num_hidden_layers
        // is still 0, so head_dim/linear_*_dim are unused decoration; only
        // hidden_size needs to satisfy the Q8_0 block-size constraint.
        const NEON_HIDDEN: usize = 32;
        let cfg = Qwen35Config {
            hidden_size: NEON_HIDDEN,
            head_dim: NEON_HIDDEN,
            linear_key_head_dim: NEON_HIDDEN,
            linear_value_head_dim: NEON_HIDDEN,
            linear_conv_kernel_dim: NEON_HIDDEN,
            ..zero_layer_config()
        };
        let embed = vec![0.0f32; VOCAB * NEON_HIDDEN];
        // Embed values are all 0.0 — finite, so packing cannot fail.
        let lm_head_packed = pack_weights_q8(&embed, VOCAB, NEON_HIDDEN)
            .expect("packing all-zero weights must succeed");
        let model = Q8NeonModel {
            embed_tokens: embed,
            final_norm: vec![0.0f32; NEON_HIDDEN],
            lm_head_packed,
            lm_head_rows: VOCAB,
            lm_head_cols: NEON_HIDDEN,
            layers: vec![],
        };
        let out = generate_q8_neon(
            &model,
            &cfg,
            &minimal_tokenizer(),
            &minimal_rope(),
            "h",
            &stop_gen_cfg(),
        )
        .expect("generate_q8_neon must succeed");
        assert_excludes_stop_token(&out, "generate_q8_neon");
    }
}
