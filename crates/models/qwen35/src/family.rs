//! Qwen3.5 CPU decoder descriptor and loaded-model adapter.

use crate::decoder::{DecoderSession, qwen_cpu::QwenCpuSession};
use crate::model::qwen35::Qwen35Model;
use crate::model::qwen35::{
    check_prompt_ids_in_vocab, check_prompt_not_empty, resolve_reasoning_close_token,
};
use lattice_arch::decoder::{DecoderModel, SessionRequest, TextDecoder, TextStream};
use lattice_arch::family::{FamilyDescriptor, LoadRequest};
use lattice_arch::generation::GenerateConfig;
use lattice_inference_core::InferenceError;
use lattice_tokenizer::__internal::{
    IncrementalDetokenizer, decode_tokens, incremental_detokenizer_finish,
    incremental_detokenizer_new, incremental_detokenizer_push_into, newline_boundary_table,
};
use lattice_tokenizer::Tokenizer;
use lattice_tokenizer::bpe::BpeTokenizer;
use std::sync::Arc;

/// A loaded Qwen3.5 model behind [`DecoderModel`], opening the CPU sessions that
/// [`Qwen35Model::generate`] runs.
pub struct Qwen35CpuDecoder {
    model: Qwen35Model,
}

impl Qwen35CpuDecoder {
    /// Wraps a loaded model.
    pub fn new(model: Qwen35Model) -> Self {
        Self { model }
    }

    /// The loaded model.
    pub fn model(&self) -> &Qwen35Model {
        &self.model
    }

    /// The loaded model, by value.
    pub fn into_model(self) -> Qwen35Model {
        self.model
    }
}

impl DecoderModel for Qwen35CpuDecoder {
    /// Tokenizes `prompt` as [`Qwen35Model::generate`] does and admits it in the same order:
    /// the empty-prompt refusal, then (unless `max_new_tokens` is 0) the context budget, then
    /// the reasoning close token and the line-ending table of the model's tokenizer.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] for a prompt with no tokens, and when the prompt's token
    /// count plus the decode cap exceeds [`Qwen35Model::max_context`].
    /// [`InferenceError::InvalidInput`] when `config` enforces a reasoning budget but the
    /// tokenizer has no single in-vocabulary `</think>` token.
    fn admit(
        &self,
        prompt: &str,
        config: GenerateConfig,
    ) -> Result<SessionRequest, InferenceError> {
        let model = &self.model;
        let cfg = model.config();
        let input = model.tokenizer().tokenize(prompt);
        let prompt_ids = input.input_ids[..input.real_length].to_vec();
        check_prompt_not_empty(prompt_ids.len())?;
        if config.max_new_tokens == 0 {
            return Ok(SessionRequest::new(prompt_ids, config, cfg.eos_token_id));
        }
        let reasoning_close = decode_admission(model, prompt_ids.len(), &config)?;
        let request = SessionRequest::new(prompt_ids, config, cfg.eos_token_id);
        Ok(match reasoning_close {
            Some((id, table)) => request.with_reasoning_close(id, table),
            None => request,
        })
    }

    fn text(&self) -> Arc<dyn TextDecoder> {
        Arc::new(BpeText {
            tokenizer: self.model.tokenizer().clone(),
        })
    }

    /// Refuses an empty prompt ([`InferenceError::Inference`]), and a prompt id outside the
    /// vocabulary or an end-of-sequence id that differs from the model's
    /// ([`InferenceError::InvalidInput`]). Unless `max_new_tokens` is 0, also refuses a prompt and
    /// decode cap past [`Qwen35Model::max_context`] ([`InferenceError::Inference`]), and a
    /// reasoning close token or line-ending table that differs from the one
    /// [`DecoderModel::admit`] gives for the request's configuration
    /// ([`InferenceError::InvalidInput`]).
    fn session<'m>(
        &'m mut self,
        request: &'m SessionRequest,
    ) -> Result<Box<dyn DecoderSession + 'm>, InferenceError> {
        let model: &'m Qwen35Model = &self.model;
        let cfg = model.config();
        let gen_cfg = request.config();
        let prompt = request.prompt_ids();
        check_prompt_not_empty(prompt.len())?;
        check_prompt_ids_in_vocab(prompt, cfg.vocab_size)?;
        if request.eos_token_id() != cfg.eos_token_id {
            return Err(InferenceError::InvalidInput(format!(
                "the request's end-of-sequence id {} differs from the model's {}",
                request.eos_token_id(),
                cfg.eos_token_id
            )));
        }
        if gen_cfg.max_new_tokens > 0 {
            let reasoning_close = decode_admission(model, prompt.len(), gen_cfg)?;
            let think_close_id = reasoning_close.as_ref().map(|(id, _)| *id);
            if request.think_close_id() != think_close_id {
                return Err(InferenceError::InvalidInput(format!(
                    "the request's reasoning close token {:?} differs from the model's {think_close_id:?}",
                    request.think_close_id()
                )));
            }
            if let Some((_, table)) = &reasoning_close
                && request.reasoning_newline_boundary() != Some(table)
            {
                return Err(InferenceError::InvalidInput(
                    "the request's line-ending table differs from the model's tokenizer".into(),
                ));
            }
        }
        let mut session =
            QwenCpuSession::new(model, prompt.to_vec(), gen_cfg.temperature, gen_cfg.seed)?;
        session.reserve_generation(prompt.len().saturating_add(gen_cfg.effective_decode_cap()));
        Ok(Box::new(session))
    }
}

/// A request's reasoning close token and the tokenizer's line-ending table.
type ReasoningClose = (u32, Arc<[bool]>);

/// The admission rule for a request with a non-zero decode cap, in [`Qwen35Model::generate`]'s
/// order: the context budget, then the reasoning close token and, when one is set, the tokenizer's
/// line-ending table. `admit` attaches the result to the request; `session` recomputes it to
/// refuse a request this model would not have admitted.
fn decode_admission(
    model: &Qwen35Model,
    prompt_len: usize,
    config: &GenerateConfig,
) -> Result<Option<ReasoningClose>, InferenceError> {
    crate::model::qwen35::check_context_budget_with_total(
        prompt_len,
        config.effective_reasoning_budget(),
        config.max_new_tokens,
        config.total_token_cap,
        model.max_context(),
    )?;
    let think_close_id = resolve_reasoning_close_token(
        model.tokenizer(),
        config.reasoning_budget,
        config.enable_thinking,
        model.config().vocab_size,
    )?;
    Ok(think_close_id.map(|id| (id, newline_boundary_table(model.tokenizer()))))
}

/// The text of a Qwen3.5 model's token ids. Holds a clone of the model's tokenizer, which is a
/// shared handle to the same vocabulary.
struct BpeText {
    tokenizer: BpeTokenizer,
}

impl TextDecoder for BpeText {
    fn decode(&self, ids: &[u32]) -> String {
        decode_tokens(&self.tokenizer, ids)
    }

    fn stream(&self) -> Box<dyn TextStream + '_> {
        Box::new(BpeStream {
            tokenizer: &self.tokenizer,
            detokenizer: incremental_detokenizer_new(),
        })
    }
}

struct BpeStream<'t> {
    tokenizer: &'t BpeTokenizer,
    detokenizer: IncrementalDetokenizer,
}

impl TextStream for BpeStream<'_> {
    fn push_into(&mut self, id: u32, out: &mut String) {
        incremental_detokenizer_push_into(&mut self.detokenizer, self.tokenizer, id, out);
    }

    fn finish(&mut self) -> String {
        incremental_detokenizer_finish(&mut self.detokenizer)
    }
}

fn load_qwen35(request: &LoadRequest) -> Result<Box<dyn DecoderModel>, InferenceError> {
    let model = Qwen35Model::from_safetensors(request.model_dir())?;
    Ok(Box::new(Qwen35CpuDecoder::new(model)))
}

/// Qwen3.5 text generation on the CPU: `model_type` `qwen3_5`, `qwen3_5_moe` or `qwen3_5_text`.
pub static QWEN35: FamilyDescriptor = FamilyDescriptor::decoder(
    "qwen35",
    &["qwen3_5", "qwen3_5_moe", "qwen3_5_text"],
    load_qwen35,
);

#[cfg(test)]
mod tests;
