//! The family contract of `lattice-arch` (ADR-103 Amendment 21 item 3) and the descriptors for
//! the model families this crate loads.
//!
//! **Unstable**: [`QWEN35`](crate::family::QWEN35) offers a Qwen3.5 decoder on the CPU through
//! [`Qwen35CpuDecoder`](crate::family::Qwen35CpuDecoder); [`BERT`](crate::family::BERT) and
//! [`EMBEDDING_GEMMA2`](crate::family::EMBEDDING_GEMMA2) offer the CPU encoders of
//! [`crate::encoder`]. [`registry`](crate::family::registry) builds a table of all three. The
//! Qwen3 embedding model has no descriptor yet: its loader chooses the Metal backend itself, so a
//! [`LoadRequest`](crate::family::LoadRequest) cannot fix its backend.

pub use lattice_arch::decoder::{DecoderModel, SessionRequest, TextDecoder, TextStream};
pub use lattice_arch::family::{
    Backend, DecoderLoader, EncoderLoader, FamilyDescriptor, LoadRequest, ModelKind, Registry,
    RegistryError, ResolveKey,
};

use crate::decoder::DecoderSession;
use crate::decoder::qwen_cpu::QwenCpuSession;
use crate::encoder::{BertEncoder, EmbeddingGemma2Encoder, Encoder};
use crate::error::InferenceError;
use crate::generation::GenerateConfig;
use crate::model::BertModel;
use crate::model::embeddinggemma2::EmbeddingGemma2Model;
use crate::model::qwen35::Qwen35Model;
use crate::model::qwen35::{
    check_context_budget, check_prompt_ids_in_vocab, check_prompt_not_empty,
    resolve_reasoning_close_token,
};
use crate::model::qwen35_config::decode_cap;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use lattice_tokenizer::__internal::{
    IncrementalDetokenizer, decode_tokens, incremental_detokenizer_finish,
    incremental_detokenizer_new, incremental_detokenizer_push_into, newline_boundary_table,
};
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
        check_context_budget(
            prompt_ids.len(),
            config.effective_reasoning_budget(),
            config.max_new_tokens,
            model.max_context(),
        )?;
        let think_close_id = resolve_reasoning_close_token(
            model.tokenizer(),
            config.reasoning_budget,
            config.enable_thinking,
            cfg.vocab_size,
        )?;
        let request = SessionRequest::new(prompt_ids, config, cfg.eos_token_id);
        Ok(match think_close_id {
            Some(id) => request.with_reasoning_close(id, newline_boundary_table(model.tokenizer())),
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
            check_context_budget(
                prompt.len(),
                gen_cfg.effective_reasoning_budget(),
                gen_cfg.max_new_tokens,
                model.max_context(),
            )?;
            let think_close_id = resolve_reasoning_close_token(
                model.tokenizer(),
                gen_cfg.reasoning_budget,
                gen_cfg.enable_thinking,
                cfg.vocab_size,
            )?;
            if request.think_close_id() != think_close_id {
                return Err(InferenceError::InvalidInput(format!(
                    "the request's reasoning close token {:?} differs from the model's {think_close_id:?}",
                    request.think_close_id()
                )));
            }
            if think_close_id.is_some() {
                let table = newline_boundary_table(model.tokenizer());
                if request.reasoning_newline_boundary() != Some(&table) {
                    return Err(InferenceError::InvalidInput(
                        "the request's line-ending table differs from the model's tokenizer".into(),
                    ));
                }
            }
        }
        let mut session =
            QwenCpuSession::new(model, prompt.to_vec(), gen_cfg.temperature, gen_cfg.seed)?;
        session.reserve_generation(prompt.len().saturating_add(decode_cap(
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
        )));
        Ok(Box::new(session))
    }
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

fn load_bert(request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
    let model = BertModel::from_directory(request.model_dir())?;
    Ok(Arc::new(BertEncoder::new(Arc::new(model))?))
}

fn load_embeddinggemma2(request: &LoadRequest) -> Result<Arc<dyn Encoder>, InferenceError> {
    let model = EmbeddingGemma2Model::from_model_dir(request.model_dir())?;
    Ok(Arc::new(EmbeddingGemma2Encoder::new(
        Arc::new(model),
        None,
    )?))
}

/// Qwen3.5 text generation on the CPU: `model_type` `qwen3_5`, `qwen3_5_moe` or `qwen3_5_text`.
pub static QWEN35: FamilyDescriptor = FamilyDescriptor::decoder(
    "qwen35",
    &["qwen3_5", "qwen3_5_moe", "qwen3_5_text"],
    load_qwen35,
);

/// BERT-family text embeddings on the CPU: `model_type` `bert`.
pub static BERT: FamilyDescriptor = FamilyDescriptor::encoder("bert", &["bert"], load_bert);

/// EmbeddingGemma 2 text embeddings on the CPU at the native width. Resolved by family name: it
/// claims no `model_type`.
pub static EMBEDDING_GEMMA2: FamilyDescriptor =
    FamilyDescriptor::encoder("embeddinggemma2", &[], load_embeddinggemma2);

/// A registry of [`QWEN35`], [`BERT`] and [`EMBEDDING_GEMMA2`].
///
/// # Errors
///
/// None in practice: the three descriptors share no name and no claim. The `Result` is
/// [`Registry::new`]'s.
pub fn registry() -> Result<Registry, RegistryError> {
    Registry::new(&[&QWEN35, &BERT, &EMBEDDING_GEMMA2])
}

#[cfg(test)]
mod tests;
