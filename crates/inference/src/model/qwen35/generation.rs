use super::cache::{ForwardScratch, KvCache};
use super::detokenize::decode_tokens;
use super::model::Qwen35Model;
use super::sampling::sample_token;
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
use crate::tokenizer::common::Tokenizer;

impl Qwen35Model {
    /// **Unstable**: autoregressive text generation with temperature/top-k/top-p sampling.
    pub fn generate(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        let cfg = &self.config;

        let mut rng_state = initial_rng_state(gen_cfg.seed);

        let input = self.tokenizer.tokenize(prompt);
        let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
        let prompt_len = prompt_ids.len();

        if prompt_len == 0 {
            return Err(InferenceError::Inference("empty prompt".into()));
        }

        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut all_ids = prompt_ids.clone();

        prefill_tokens(
            self,
            &prompt_ids,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        kv_cache.seq_len = prompt_len;

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            &all_ids,
            &mut rng_state,
        );

        if should_stop_token(cfg, gen_cfg, next_id) {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
            });
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);

        decode_loop(
            self,
            gen_cfg,
            &mut all_ids,
            &mut generated_ids,
            &mut rng_state,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );

        let text = decode_tokens(&self.tokenizer, &generated_ids);

        Ok(GenerateOutput {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
        })
    }
}

fn initial_rng_state(seed: Option<u64>) -> u64 {
    match seed {
        Some(s) => {
            if s == 0 {
                1
            } else {
                s
            }
        }
        None => {
            use std::time::SystemTime;
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678_9abcdef0);
            if t == 0 { 1 } else { t }
        }
    }
}

fn prefill_tokens(
    model: &Qwen35Model,
    prompt_ids: &[u32],
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
) {
    let prompt_len = prompt_ids.len();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        model.forward_step(token_id, pos, gdn_states, kv_cache, scratch);
        if pos < prompt_len - 1 {
            kv_cache.seq_len += 1;
        }
    }
}

fn decode_loop(
    model: &Qwen35Model,
    gen_cfg: &GenerateConfig,
    all_ids: &mut Vec<u32>,
    generated_ids: &mut Vec<u32>,
    rng_state: &mut u64,
    gdn_states: &mut [GatedDeltaNetState],
    kv_cache: &mut KvCache,
    scratch: &mut ForwardScratch,
) {
    let cfg = &model.config;
    for _ in 1..gen_cfg.max_new_tokens {
        let pos = kv_cache.seq_len;
        let last_token = *all_ids
            .last()
            .expect("invariant: prompt or prior generation seeded all_ids");

        model.forward_step(last_token, pos, gdn_states, kv_cache, scratch);
        kv_cache.seq_len += 1;

        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            gen_cfg,
            all_ids,
            rng_state,
        );

        if should_stop_token(cfg, gen_cfg, next_id) {
            break;
        }

        generated_ids.push(next_id);
        all_ids.push(next_id);
    }
}

/// Returns true when `token_id` is EOS or is in the `stop_token_ids` list.
pub fn should_stop_token(cfg: &Qwen35Config, gen_cfg: &GenerateConfig, token_id: u32) -> bool {
    token_id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&token_id)
}
