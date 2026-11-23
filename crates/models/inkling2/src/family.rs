//! The loaded model and its tokenizer behind `lattice_arch::decoder::DecoderModel`, its
//! generation entry points, and the family descriptor for `model_type` `inkling2_loop`.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use lattice_arch::__internal::check_prompt_not_empty;
use lattice_arch::decoder::{
    Cancellation, DecoderModel, DecoderSession, SessionRequest, TextDecoder, TextStream, driver,
};
use lattice_arch::family::{FamilyDescriptor, LoadRequest};
use lattice_arch::generation::{GenerateConfig, GenerateOutput};
use lattice_arch::inkling2_config::parse_config;
use lattice_arch::stop_reason::StopReason;
use lattice_inference_core::InferenceError;
use lattice_tokenizer::__internal::{
    IncrementalDetokenizer, decode_tokens, incremental_detokenizer_finish,
    incremental_detokenizer_new, incremental_detokenizer_push_into,
    tokenize_fragments_with_inserted_ids,
};
use lattice_tokenizer::BpeTokenizer;

use crate::loader::Inkling2Checkpoint;
use crate::model::Inkling2Model;
use crate::session::Inkling2CpuSession;
use crate::weights::Inkling2Weights;

/// An Inkling 2 model with its tokenizer and end-of-sequence id, generating on the CPU.
pub struct Inkling2CpuDecoder {
    model: Inkling2Model,
    tokenizer: BpeTokenizer,
    eos_token_id: u32,
}

impl Inkling2CpuDecoder {
    /// Pairs a model with its tokenizer and end-of-sequence id.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] when `eos_token_id` is outside the model's unpadded
    /// vocabulary.
    pub fn new(
        model: Inkling2Model,
        tokenizer: BpeTokenizer,
        eos_token_id: u32,
    ) -> Result<Self, InferenceError> {
        let vocab = model.config().unpadded_vocab_size;
        if eos_token_id as usize >= vocab {
            return Err(InferenceError::InvalidInput(format!(
                "end-of-sequence id {eos_token_id} is outside the vocabulary of {vocab} tokens"
            )));
        }
        Ok(Self {
            model,
            tokenizer,
            eos_token_id,
        })
    }

    /// Loads `config.json`, the safetensors weights and `tokenizer.json` from a checkpoint
    /// directory. The end-of-sequence id is the configuration's top-level `eos_token_id`.
    ///
    /// # Errors
    ///
    /// The errors of the configuration parser, [`Inkling2Checkpoint::open`],
    /// [`Inkling2Weights::load`], [`Inkling2Model::new`], the tokenizer loader and
    /// [`Inkling2CpuDecoder::new`].
    pub fn open(dir: &Path) -> Result<Self, InferenceError> {
        let config = parse_config(&dir.join("config.json"))?;
        let eos_token_id = u32::try_from(config.eos_token_id).map_err(|_| {
            InferenceError::InvalidInput(format!(
                "end-of-sequence id {} does not fit a token id",
                config.eos_token_id
            ))
        })?;
        let weights = Inkling2Weights::load(&Inkling2Checkpoint::open(dir)?, &config.text)?;
        let model = Inkling2Model::new(config.text, weights)?;
        let tokenizer = BpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?;
        Self::new(model, tokenizer, eos_token_id)
    }

    /// The model.
    pub fn model(&self) -> &Inkling2Model {
        &self.model
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// The end-of-sequence token id.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// The token ids of `text`. A special token written in the text, such as
    /// `<|message_user|>`, becomes its single id. Nothing is truncated or padded.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        tokenize_fragments_with_inserted_ids(&self.tokenizer, text, &[], "")
    }

    /// Generates from `prompt_ids` until the end-of-sequence id, a stop id of `gen_cfg` or its
    /// decode cap. See [`Inkling2CpuDecoder::generate_streaming_with_cancel`].
    ///
    /// # Errors
    ///
    /// As [`Inkling2CpuDecoder::generate_streaming_with_cancel`].
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> {
        self.generate_streaming_with_cancel(prompt_ids, gen_cfg, |_| true, || false)
    }

    /// Generates from `prompt_ids`, passing each piece of text to `on_token` as it is
    /// confirmed. `should_cancel` is polled before the prefill, between prefill chunks and
    /// before each decode step; it, or `on_token` returning `false`, ends generation with
    /// [`StopReason::Interrupt`]. Special tokens decode to no text, so the returned text joins
    /// the reasoning and the answer; the token ids keep them apart. The end-of-sequence and stop
    /// ids are not part of the output.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Inference`] for an empty prompt and for a prompt whose length plus the
    /// decode cap exceeds `max_position_embeddings`; [`InferenceError::InvalidInput`] for a
    /// prompt id outside the vocabulary and for a control the session does not support (a
    /// grammar, log-probabilities, stop strings or a reasoning budget); an error of the
    /// forward pass.
    pub fn generate_streaming_with_cancel<F, C>(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
        mut on_token: F,
        should_cancel: C,
    ) -> Result<GenerateOutput, InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        let prompt_tokens = prompt_ids.len();
        check_prompt_not_empty(prompt_tokens)?;
        self.check_prompt_ids(prompt_ids)?;
        if gen_cfg.max_new_tokens == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: Vec::new(),
                prompt_tokens,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: Vec::new(),
            });
        }
        self.check_context(prompt_tokens, gen_cfg)?;
        let mut session = Inkling2CpuSession::new(
            &self.model,
            prompt_ids.to_vec(),
            gen_cfg.temperature,
            gen_cfg.seed,
        )?;
        let should_cancel = RefCell::new(should_cancel);
        let cancel = FnMutCancellation(&should_cancel);
        let stream = RefCell::new(BpeStream {
            tokenizer: &self.tokenizer,
            detokenizer: incremental_detokenizer_new(),
        });
        let mut text = String::new();
        let mut token_logprob_end_offsets = Vec::new();
        let result = driver::run_into(
            &mut session,
            gen_cfg,
            None,
            None,
            prompt_ids,
            self.eos_token_id,
            true,
            &cancel,
            |_| {},
            |id, out| stream.borrow_mut().push_into(id, out),
            &mut text,
            &mut token_logprob_end_offsets,
            |delta, _| on_token(delta),
            || {},
            || stream.borrow_mut().finish(),
        )?;
        Ok(GenerateOutput {
            text,
            generated_tokens: result.generated_ids.len(),
            token_ids: result.generated_ids,
            prompt_tokens,
            stopped: result.stopped,
            stop_reason: Some(result.stop_reason),
            token_logprobs: result.token_logprobs,
        })
    }

    fn check_prompt_ids(&self, prompt_ids: &[u32]) -> Result<(), InferenceError> {
        let vocab = self.model.config().unpadded_vocab_size;
        match prompt_ids
            .iter()
            .enumerate()
            .find(|(_, id)| **id as usize >= vocab)
        {
            Some((index, id)) => Err(InferenceError::InvalidInput(format!(
                "prompt token {id} at position {index} is outside the vocabulary of {vocab} tokens"
            ))),
            None => Ok(()),
        }
    }

    /// The admission rule for a request that generates: the prompt and the decode cap fit the
    /// context, and no reasoning budget is asked for, since the model has no single token that
    /// closes its reasoning.
    fn check_decode(
        &self,
        prompt_len: usize,
        config: &GenerateConfig,
    ) -> Result<(), InferenceError> {
        self.check_context(prompt_len, config)?;
        if config.effective_reasoning_budget().is_some() {
            return Err(InferenceError::InvalidInput(
                "a reasoning budget cannot be enforced: Inkling 2 has no single token that \
                 closes its reasoning"
                    .into(),
            ));
        }
        Ok(())
    }

    fn check_context(
        &self,
        prompt_len: usize,
        config: &GenerateConfig,
    ) -> Result<(), InferenceError> {
        let cap = config.effective_decode_cap();
        let window = self.model.config().max_position_embeddings;
        if prompt_len.saturating_add(cap) > window {
            return Err(InferenceError::Inference(format!(
                "prompt ({prompt_len} tokens) plus decode cap ({cap} tokens) exceeds the \
                 Inkling 2 context window ({window})"
            )));
        }
        Ok(())
    }
}

impl DecoderModel for Inkling2CpuDecoder {
    /// Tokenizes `prompt` as [`Inkling2CpuDecoder::tokenize`] does and admits it: the
    /// empty-prompt refusal, then, unless `max_new_tokens` is 0, the context budget and the
    /// reasoning-budget refusal. The request carries no reasoning close token.
    fn admit(
        &self,
        prompt: &str,
        config: GenerateConfig,
    ) -> Result<SessionRequest, InferenceError> {
        let prompt_ids = self.tokenize(prompt);
        check_prompt_not_empty(prompt_ids.len())?;
        if config.max_new_tokens > 0 {
            self.check_decode(prompt_ids.len(), &config)?;
        }
        Ok(SessionRequest::new(prompt_ids, config, self.eos_token_id))
    }

    fn text(&self) -> Arc<dyn TextDecoder> {
        Arc::new(BpeText {
            tokenizer: self.tokenizer.clone(),
        })
    }

    /// Refuses an empty prompt and, unless `max_new_tokens` is 0, a prompt and decode cap past
    /// the context window ([`InferenceError::Inference`]); a prompt id outside the vocabulary, an
    /// end-of-sequence id other than the model's, a reasoning close token, and a reasoning budget
    /// ([`InferenceError::InvalidInput`]).
    fn session<'m>(
        &'m mut self,
        request: &'m SessionRequest,
    ) -> Result<Box<dyn DecoderSession + 'm>, InferenceError> {
        let prompt = request.prompt_ids();
        let config = request.config();
        check_prompt_not_empty(prompt.len())?;
        self.check_prompt_ids(prompt)?;
        if request.eos_token_id() != self.eos_token_id {
            return Err(InferenceError::InvalidInput(format!(
                "the request's end-of-sequence id {} differs from the model's {}",
                request.eos_token_id(),
                self.eos_token_id
            )));
        }
        if let Some(id) = request.think_close_id() {
            return Err(InferenceError::InvalidInput(format!(
                "the request names reasoning close token {id}; Inkling 2 has none"
            )));
        }
        if config.max_new_tokens > 0 {
            self.check_decode(prompt.len(), config)?;
        }
        let session = Inkling2CpuSession::new(
            &self.model,
            prompt.to_vec(),
            config.temperature,
            config.seed,
        )?;
        Ok(Box::new(session))
    }
}

/// Adapts a polled `FnMut() -> bool` to [`Cancellation`], whose blanket implementation covers
/// only `Fn` closures.
struct FnMutCancellation<'a, F: FnMut() -> bool>(&'a RefCell<F>);

impl<F: FnMut() -> bool> Cancellation for FnMutCancellation<'_, F> {
    fn is_cancelled(&self) -> bool {
        (self.0.borrow_mut())()
    }
}

/// The text of a model's token ids, holding a clone of the tokenizer, which shares its
/// vocabulary.
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

fn load(request: &LoadRequest) -> Result<Box<dyn DecoderModel>, InferenceError> {
    Ok(Box::new(Inkling2CpuDecoder::open(request.model_dir())?))
}

/// Inkling 2 text generation on the CPU: `model_type` `inkling2_loop`.
pub static INKLING2: FamilyDescriptor =
    FamilyDescriptor::decoder("inkling2", &["inkling2_loop"], load);
