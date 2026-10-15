//! Model-derived reference evidence for admission; no caller-supplied NLL values.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::statistics::AdmissionContract;
use crate::router_refit::{RefitCandidate, RefitSource};

/// One operator-provided single-user prompt and its teacher-forced completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceExample {
    /// Raw user text, also used verbatim as the routing embedder input.
    pub user_prompt: String,
    /// Completion text following the rendered assistant-turn opener.
    pub completion: String,
}

/// Explicit work limits; neither deserialization nor construction fills numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceBounds {
    /// Maximum number of reference examples.
    pub max_examples: usize,
    /// Maximum cumulative prompt-plus-completion tokens, counted once across arms.
    pub max_tokens: usize,
}

/// Read-only identity of one exact adapter payload used by the scorer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReferencePayloadSummary {
    /// Adapter name in trained output-column order.
    pub name: String,
    /// Hash of descriptor, rotation declaration and every tensor bit.
    pub content_hash: String,
    /// Descriptor rank, checked against every layer.
    pub rank: usize,
    /// Exact f32 alpha bits; effective scale is the validated alpha/rank.
    pub alpha_bits: u32,
    /// Declared rotation seed, checked against the actual base state.
    pub quarot_seed: Option<u64>,
    /// Number of distinct layer/module projections.
    pub layers: usize,
}

/// Evidence produced only by the concrete model scorer, never deserialized.
#[derive(Debug, Clone, Serialize)]
pub struct ReferenceEvidence {
    pub(super) contract_hash: String,
    pub(super) parent_artifact_hash: String,
    pub(super) parent_gate_hash: String,
    pub(super) candidate_gate_hash: String,
    pub(super) policy_hash: String,
    pub(super) reference_hash: String,
    pub(super) payload_hash: String,
    pub(super) payload_summary: Vec<ReferencePayloadSummary>,
    pub(super) execution_hash: String,
    pub(super) binding_hash: String,
    pub(super) token_count: usize,
    pub(super) parent_mean_nll: f64,
    pub(super) candidate_mean_nll: f64,
    pub(super) base_mean_nll: f64,
    pub(super) regression: f64,
}

/// Missing, unsupported or invalid evidence refuses instead of supplying a score.
#[derive(Debug, thiserror::Error)]
pub enum ReferenceError {
    /// The source, payload, reference or explicit bounds are inconsistent.
    #[error("invalid reference scoring input: {0}")]
    Invalid(String),
    /// The real embedder, router or model refused execution.
    #[error("reference model execution failed: {0}")]
    Model(String),
    /// A count overflowed or cannot be represented exactly.
    #[error("unrepresentable reference quantity: {0}")]
    Count(&'static str),
    /// Full-vocabulary logits or derived NLL were not finite and usable.
    #[error("invalid reference numerical evidence: {0}")]
    Numerical(&'static str),
}

fn invalid(message: impl Into<String>) -> ReferenceError {
    ReferenceError::Invalid(message.into())
}

fn framed(hash: &mut Sha256, bytes: &[u8]) -> Result<(), ReferenceError> {
    let len = u64::try_from(bytes.len()).map_err(|_| ReferenceError::Count("hash length"))?;
    hash.update(len.to_be_bytes());
    hash.update(bytes);
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn serialized_hash(domain: &[u8], value: &impl Serialize) -> Result<String, ReferenceError> {
    let bytes = serde_json::to_vec(value).map_err(|e| invalid(e.to_string()))?;
    let mut hash = Sha256::new();
    hash.update(domain);
    framed(&mut hash, &bytes)?;
    Ok(format!("{:x}", hash.finalize()))
}

/// Hash ordered raw prompt/completion strings before scoring or choosing an outcome.
pub fn reference_content_hash(examples: &[ReferenceExample]) -> Result<String, ReferenceError> {
    if examples.is_empty() {
        return Err(invalid("reference set is empty"));
    }
    let mut hash = Sha256::new();
    hash.update(b"lattice-router-reference-content-v1\0");
    for example in examples {
        if example.user_prompt.is_empty() || example.completion.is_empty() {
            return Err(invalid("reference prompt and completion must be nonempty"));
        }
        framed(&mut hash, example.user_prompt.as_bytes())?;
        framed(&mut hash, example.completion.as_bytes())?;
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn policy_hash(source: &RefitSource) -> Result<String, ReferenceError> {
    serialized_hash(
        b"lattice-router-reference-policy-v1\0",
        &(
            source.policy(),
            source.representation(),
            source.base_model(),
            source.adapter_names(),
        ),
    )
}

impl ReferenceEvidence {
    #[cfg(all(test, feature = "serve"))]
    pub(super) fn fixture(
        source: &RefitSource,
        candidate: &RefitCandidate,
        contract: &AdmissionContract,
        reference_hash: &str,
        token_count: usize,
        nlls: [f64; 3],
    ) -> Self {
        let payload_summary = vec![];
        let mut evidence = Self {
            contract_hash: contract.hash().unwrap(),
            parent_artifact_hash: source.artifact_hash().to_string(),
            parent_gate_hash: digest(source.gate_bytes()),
            candidate_gate_hash: digest(candidate.gate_bytes()),
            policy_hash: policy_hash(source).unwrap(),
            reference_hash: reference_hash.to_string(),
            payload_hash: serialized_hash(
                b"lattice-router-reference-payload-set-v1\0",
                &payload_summary,
            )
            .unwrap(),
            payload_summary,
            execution_hash: digest(b"test-only reference execution fixture"),
            binding_hash: String::new(),
            token_count,
            parent_mean_nll: nlls[0],
            candidate_mean_nll: nlls[1],
            base_mean_nll: nlls[2],
            regression: nlls[1] - nlls[0],
        };
        evidence.binding_hash = evidence.compute_binding_hash().unwrap();
        evidence
    }

    /// The operator-predeclared content identity this evidence actually scored.
    pub fn reference_hash(&self) -> &str {
        &self.reference_hash
    }

    /// Check the immutable evidence's producing source, candidate and contract.
    pub(super) fn matches(
        &self,
        source: &RefitSource,
        candidate: &RefitCandidate,
        contract: &AdmissionContract,
    ) -> Result<(), ReferenceError> {
        let contract_hash = contract.hash().map_err(|e| invalid(e.to_string()))?;
        if !candidate.matches_source(source)
            || candidate.admission_contract_hash() != Some(contract_hash.as_str())
            || self.contract_hash != contract_hash
            || self.parent_artifact_hash != source.artifact_hash()
            || self.parent_gate_hash != digest(source.gate_bytes())
            || self.candidate_gate_hash != digest(candidate.gate_bytes())
            || self.policy_hash != policy_hash(source)?
            || self.binding_hash != self.compute_binding_hash()?
        {
            return Err(invalid(
                "reference evidence does not bind this candidate/source/contract",
            ));
        }
        Ok(())
    }

    fn compute_binding_hash(&self) -> Result<String, ReferenceError> {
        serialized_hash(
            b"lattice-router-reference-evidence-v1\0",
            &(
                &self.contract_hash,
                &self.parent_artifact_hash,
                &self.parent_gate_hash,
                &self.candidate_gate_hash,
                &self.policy_hash,
                &self.reference_hash,
                &self.payload_hash,
                &self.payload_summary,
                &self.execution_hash,
                self.token_count,
                self.parent_mean_nll.to_bits(),
                self.candidate_mean_nll.to_bits(),
                self.base_mean_nll.to_bits(),
                self.regression.to_bits(),
            ),
        )
    }
}

#[cfg(any(
    test,
    all(
        feature = "serve",
        feature = "f16",
        target_os = "macos",
        feature = "metal-gpu"
    )
))]
mod scoring {
    use super::*;

    trait TeacherForcer {
        fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>, ReferenceError>;
        fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>, ReferenceError>;
    }

    fn completion_nll(
        model: &mut impl TeacherForcer,
        prompt: &[u32],
        completion: &[u32],
        vocab: usize,
    ) -> Result<f64, ReferenceError> {
        if prompt.is_empty() || completion.is_empty() || vocab == 0 {
            return Err(invalid("empty prompt, completion or vocabulary"));
        }
        prompt
            .len()
            .checked_add(completion.len())
            .ok_or(ReferenceError::Count("token range"))?;
        if prompt
            .iter()
            .chain(completion)
            .any(|&token| token as usize >= vocab)
        {
            return Err(invalid("reference token is outside the model vocabulary"));
        }
        let mut logits = model.prefill(prompt)?;
        let mut total = 0.0;
        for (index, &target) in completion.iter().enumerate() {
            total += full_vocab_nll(&logits, target, vocab)?;
            if !total.is_finite() {
                return Err(ReferenceError::Numerical("NLL sum"));
            }
            if index + 1 < completion.len() {
                let position = prompt
                    .len()
                    .checked_add(index)
                    .ok_or(ReferenceError::Count("position"))?;
                logits = model.step(target, position)?;
            }
        }
        Ok(total)
    }

    fn full_vocab_nll(logits: &[f32], target: u32, vocab: usize) -> Result<f64, ReferenceError> {
        if logits.len() != vocab || target as usize >= vocab || logits.is_empty() {
            return Err(invalid("logits do not contain the full model vocabulary"));
        }
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(ReferenceError::Numerical("nonfinite vocabulary logit"));
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = logits.iter().map(|&x| (f64::from(x) - max).exp()).sum();
        let nll = (max - f64::from(logits[target as usize])) + sum.ln();
        if !nll.is_finite() || nll < 0.0 {
            return Err(ReferenceError::Numerical("full-vocabulary NLL"));
        }
        Ok(nll)
    }

    fn check_bounds(
        bounds: ReferenceBounds,
        count: usize,
        tokens: usize,
    ) -> Result<(), ReferenceError> {
        if bounds.max_examples == 0
            || bounds.max_tokens == 0
            || count == 0
            || count > bounds.max_examples
            || tokens > bounds.max_tokens
        {
            return Err(invalid(
                "reference population exceeds explicit positive bounds",
            ));
        }
        if tokens as u128 > (1_u128 << 53) {
            return Err(ReferenceError::Count("exact token count"));
        }
        Ok(())
    }

    fn completion_tokens<'a>(prompt: &[u32], full: &'a [u32]) -> Result<&'a [u32], ReferenceError> {
        if prompt.is_empty() {
            return Err(invalid("reference prompt has no tokens"));
        }
        full.strip_prefix(prompt)
            .filter(|tail| !tail.is_empty())
            .ok_or_else(|| invalid("completion does not preserve the prompt token boundary"))
    }

    #[cfg(all(
        feature = "serve",
        feature = "f16",
        target_os = "macos",
        feature = "metal-gpu"
    ))]
    pub(super) mod metal {
        use super::*;
        use crate::forward::metal_qwen35::{
            ChatMessage, LoraLayerData, MetalQwen35State, blend_lora_layer_data,
            format_chat_template,
        };
        use crate::serve::embeddings::{EmbeddingModel, QWEN35_F16_DECODER_LOADER};
        use crate::serve::routing::{EmbedderIdentity, PromptSource, trained_pooling};
        use crate::tokenizer::{Tokenizer, bpe::BpeTokenizer};
        use lattice_fann::lora::LoraDescriptor;
        use std::collections::HashSet;

        /// Exact resident payload and descriptor, in trained adapter-column order.
        pub struct ReferenceAdapter<'a> {
            /// Trained adapter name.
            pub name: &'a str,
            /// Unrotated A/B payload passed through the serving blend/load path.
            pub layers: &'a [LoraLayerData],
            /// Validated descriptor supplying alpha/rank and target-module identity.
            pub descriptor: &'a LoraDescriptor,
            /// Optional declared QuaRot seed, resolved against the actual base.
            pub quarot_seed: Option<u64>,
        }

        struct Arm<'a>(&'a mut MetalQwen35State);
        impl Drop for Arm<'_> {
            fn drop(&mut self) {
                self.0.unload_lora_adapter();
                self.0.reset_state();
            }
        }
        impl TeacherForcer for Arm<'_> {
            fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>, ReferenceError> {
                self.0
                    .try_forward_prefill(prompt)
                    .map_err(|e| ReferenceError::Model(e.to_string()))
            }
            fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>, ReferenceError> {
                self.0
                    .try_forward_step(token, position)
                    .map_err(|e| ReferenceError::Model(e.to_string()))
            }
        }

        fn payloads(
            state: &MetalQwen35State,
            source: &RefitSource,
            adapters: &[ReferenceAdapter<'_>],
        ) -> Result<Vec<ReferencePayloadSummary>, ReferenceError> {
            use crate::quant::quarot::plan::{AbsorptionSide, RotationPlan};
            let cfg = &state.engine.config;
            if adapters.len() != source.adapter_names().len() || adapters.is_empty() {
                return Err(invalid(
                    "payload set differs from the trained adapter schema",
                ));
            }
            let plan = RotationPlan::qwen35_residual_stream_linear_layers();
            let mut summaries = Vec::new();
            for (adapter, name) in adapters.iter().zip(source.adapter_names()) {
                if adapter.name != name || adapter.layers.is_empty() || adapter.descriptor.rank == 0
                {
                    return Err(invalid("empty/misordered adapter or zero descriptor rank"));
                }
                adapter.descriptor.validate().map_err(invalid)?;
                lattice_fann::lora::validate_target_modules(
                    &adapter.descriptor.target_modules,
                    lattice_fann::lora::KNOWN_LORA_TARGET_MODULES,
                )
                .map_err(invalid)?;
                if adapter.quarot_seed.is_some()
                    && adapter.quarot_seed != state.engine.base_quarot_seed
                {
                    return Err(invalid(
                        "adapter rotation seed does not match the base model",
                    ));
                }
                let mut modules: Vec<_> = adapter
                    .layers
                    .iter()
                    .map(|layer| layer.module.as_str())
                    .collect();
                modules.sort_unstable();
                modules.dedup();
                let mut declared: Vec<_> = adapter
                    .descriptor
                    .target_modules
                    .iter()
                    .map(String::as_str)
                    .collect();
                declared.sort_unstable();
                declared.dedup();
                if modules != declared {
                    return Err(invalid("adapter modules differ from the descriptor"));
                }
                let mut hash = Sha256::new();
                hash.update(b"lattice-router-reference-payload-v1\0");
                framed(&mut hash, adapter.name.as_bytes())?;
                framed(&mut hash, adapter.descriptor.dtype.as_bytes())?;
                framed(&mut hash, &adapter.descriptor.alpha.to_bits().to_be_bytes())?;
                framed(
                    &mut hash,
                    &serde_json::to_vec(&(
                        adapter.descriptor.rank,
                        &adapter.descriptor.target_modules,
                        adapter.quarot_seed,
                    ))
                    .map_err(|e| invalid(e.to_string()))?,
                )?;
                let mut seen = HashSet::new();
                for layer in adapter.layers {
                    if layer.rank != adapter.descriptor.rank
                        || layer.d_in == 0
                        || layer.d_out == 0
                        || !seen.insert((layer.layer_idx, layer.module.as_str()))
                        || [layer.rank, layer.d_in, layer.d_out]
                            .iter()
                            .any(|&n| u32::try_from(n).is_err())
                    {
                        return Err(invalid("invalid or duplicate adapter projection"));
                    }
                    if crate::lora_hook::GDN_LORA_MODULES.contains(&layer.module.as_str())
                        && !crate::lora_hook::gdn_lora_module_is_servable(&layer.module)
                    {
                        return Err(invalid("adapter projection is unsupported by Metal"));
                    }
                    let shape = crate::lora_hook::qwen35_projection_shape(
                        cfg,
                        layer.layer_idx,
                        &layer.module,
                    )
                    .map_err(invalid)?;
                    if (layer.d_in, layer.d_out) != (shape.d_in, shape.d_out)
                        || layer.rank.checked_mul(layer.d_in) != Some(layer.a.len())
                        || layer.rank.checked_mul(layer.d_out) != Some(layer.b.len())
                        || layer.a.iter().chain(&layer.b).any(|x| !x.is_finite())
                    {
                        return Err(invalid(
                            "adapter payload shape or finite-value validation failed",
                        ));
                    }
                    if state.engine.base_quarot_seed.is_some() {
                        match plan.absorption_for_module(&layer.module) {
                            Some(AbsorptionSide::InputSide) if layer.d_in == cfg.hidden_size => {}
                            Some(AbsorptionSide::OutputSide) if layer.d_out == cfg.hidden_size => {}
                            _ => {
                                return Err(invalid(
                                    "adapter is incompatible with the base rotation plan",
                                ));
                            }
                        }
                    }
                    framed(
                        &mut hash,
                        &serde_json::to_vec(&(
                            layer.layer_idx,
                            &layer.module,
                            layer.rank,
                            layer.d_in,
                            layer.d_out,
                        ))
                        .map_err(|e| invalid(e.to_string()))?,
                    )?;
                    for data in [&layer.a, &layer.b] {
                        framed(&mut hash, &(data.len() as u64).to_be_bytes())?;
                        for value in data {
                            hash.update(value.to_bits().to_be_bytes());
                        }
                    }
                }
                summaries.push(ReferencePayloadSummary {
                    name: name.clone(),
                    content_hash: format!("{:x}", hash.finalize()),
                    rank: adapter.descriptor.rank,
                    alpha_bits: adapter.descriptor.alpha.to_bits(),
                    quarot_seed: adapter.quarot_seed,
                    layers: adapter.layers.len(),
                });
            }
            lattice_fann::lora::plan_blend(
                "router_reference",
                adapters.iter().flat_map(|adapter| {
                    adapter
                        .layers
                        .iter()
                        .map(|layer| lattice_fann::lora::BlendProjection {
                            layer_idx: layer.layer_idx,
                            module: &layer.module,
                            rank: layer.rank,
                            d_in: layer.d_in,
                            d_out: layer.d_out,
                        })
                }),
            )
            .map_err(invalid)?;
            Ok(summaries)
        }

        fn token_ids(tokenizer: &BpeTokenizer, text: &str) -> Result<Vec<u32>, ReferenceError> {
            let tokens = tokenizer.tokenize(text);
            if tokens.pre_truncation_len != tokens.real_length
                || tokens.real_length > tokens.input_ids.len()
            {
                return Err(invalid("reference tokenization truncated"));
            }
            Ok(tokens.input_ids[..tokens.real_length].to_vec())
        }

        fn mixture_inputs<'a>(
            adapters: &[ReferenceAdapter<'a>],
            mixture: &[(String, f32)],
        ) -> Result<Vec<(&'a [LoraLayerData], f32)>, ReferenceError> {
            if mixture.len() > adapters.len() {
                return Err(invalid(
                    "floor-applied mixture exceeds the trained adapter population",
                ));
            }
            let mut seen = HashSet::new();
            mixture
                .iter()
                .map(|(name, weight)| {
                    let adapter = adapters
                        .iter()
                        .find(|adapter| adapter.name == name)
                        .ok_or_else(|| invalid("router selected an unknown adapter"))?;
                    let scale = *weight * adapter.descriptor.scale();
                    if !seen.insert(name)
                        || !weight.is_finite()
                        || *weight < 0.0
                        || !scale.is_finite()
                    {
                        return Err(invalid(
                            "duplicate mixture member, invalid weight or nonfinite effective scale",
                        ));
                    }
                    if adapter
                        .layers
                        .iter()
                        .any(|layer| layer.b.iter().any(|&value| !(value * scale).is_finite()))
                    {
                        return Err(ReferenceError::Numerical("scaled adapter tensor"));
                    }
                    Ok((adapter.layers, scale))
                })
                .collect()
        }

        fn run_arm(
            arm: &mut Arm<'_>,
            adapters: &[ReferenceAdapter<'_>],
            mixture: &[(String, f32)],
            prompt: &[u32],
            completion: &[u32],
            vocab: usize,
        ) -> Result<f64, ReferenceError> {
            let layers = if mixture.is_empty() {
                None
            } else {
                let inputs = mixture_inputs(adapters, mixture)?;
                let layers = blend_lora_layer_data(&inputs)
                    .map_err(|e| ReferenceError::Model(e.to_string()))?;
                Some(layers)
            };
            arm.0.unload_lora_adapter();
            arm.0.reset_state();
            if let Some(layers) = layers {
                arm.0
                    .load_lora_adapter(layers, 1.0, None)
                    .map_err(|e| ReferenceError::Model(e.to_string()))?;
            }
            completion_nll(arm, prompt, completion, vocab)
        }

        /// Score three real model arms on a dedicated, initially adapter-free state.
        ///
        /// The embedder/base identities follow serving's operator-supplied names;
        /// they are not cryptographic attestations of checkpoint bytes. Every payload
        /// bit is hashed. The state is reset and unloaded on success or any fallible
        /// scoring exit. An already loaded adapter is refused without mutation.
        pub fn score_metal_reference(
            source: &RefitSource,
            candidate: &RefitCandidate,
            contract: &AdmissionContract,
            examples: &[ReferenceExample],
            bounds: ReferenceBounds,
            state: &mut MetalQwen35State,
            tokenizer: &BpeTokenizer,
            embedder: &EmbeddingModel,
            embedder_identity: &EmbedderIdentity,
            adapters: &[ReferenceAdapter<'_>],
            base_model_identity: &str,
        ) -> Result<ReferenceEvidence, ReferenceError> {
            if state.has_lora_adapter() {
                return Err(invalid(
                    "reference scorer requires a dedicated adapter-free state",
                ));
            }
            let contract_hash = contract.hash().map_err(|e| invalid(e.to_string()))?;
            if !candidate.matches_source(source)
                || candidate.admission_contract_hash() != Some(contract_hash.as_str())
                || base_model_identity != source.base_model()
            {
                return Err(invalid(
                    "candidate, contract or base identity differs from source",
                ));
            }
            check_bounds(bounds, examples.len(), 0)?;
            let reference_hash = reference_content_hash(examples)?;
            let repr = source.representation();
            if repr.embedding_model != embedder_identity.as_str()
                || repr.loader_format != QWEN35_F16_DECODER_LOADER
                || repr.prompt_source != PromptSource::SERVED.as_str()
                || repr.input_width != embedder.dimensions() as u64
            {
                return Err(invalid(
                    "embedder representation differs from registered serving representation",
                ));
            }
            let pooling = trained_pooling(&repr.pooling).map_err(|e| invalid(e.message()))?;
            let payload_summary = payloads(state, source, adapters)?;
            let payload_hash = serialized_hash(
                b"lattice-router-reference-payload-set-v1\0",
                &payload_summary,
            )?;
            let mut parent_router = source
                .policy_router(source.gate_bytes())
                .map_err(|e| invalid(e.to_string()))?;
            let mut candidate_router = source
                .policy_router(candidate.gate_bytes())
                .map_err(|e| invalid(e.to_string()))?;
            let mut execution = Sha256::new();
            execution.update(b"lattice-router-reference-execution-v1\0");
            framed(
                &mut execution,
                &serde_json::to_vec(&(
                    bounds,
                    state.engine.config.vocab_size,
                    state.max_context(),
                    state.engine.base_quarot_seed,
                ))
                .map_err(|e| invalid(e.to_string()))?,
            )?;
            let mut prepared = Vec::new();
            let mut work_tokens = 0usize;
            let mut token_count = 0usize;
            for example in examples {
                // The serving embedder exposes only post-truncation token length.
                // A byte bound plus its possible BOS/EOS bounds the BPE input safely.
                if example
                    .user_prompt
                    .len()
                    .checked_add(2)
                    .is_none_or(|n| n > embedder.max_context())
                {
                    return Err(invalid(
                        "raw reference prompt exceeds conservative embedder byte bound",
                    ));
                }
                let rendered = format_chat_template(&[ChatMessage::user(&example.user_prompt)]);
                let prompt = token_ids(tokenizer, &rendered)?;
                let full = token_ids(tokenizer, &(rendered + &example.completion))?;
                let completion = completion_tokens(&prompt, &full)?.to_vec();
                if prompt.is_empty() || full.len() > state.max_context() {
                    return Err(invalid(
                        "reference sequence is empty or exceeds model context",
                    ));
                }
                work_tokens = work_tokens
                    .checked_add(full.len())
                    .ok_or(ReferenceError::Count("work tokens"))?;
                token_count = token_count
                    .checked_add(completion.len())
                    .ok_or(ReferenceError::Count("reference tokens"))?;
                check_bounds(bounds, examples.len(), work_tokens)?;
                let embedding = embedder
                    .embed_text(&example.user_prompt, pooling)
                    .map_err(|e| ReferenceError::Model(e.to_string()))?;
                if embedding.len() != embedder.dimensions()
                    || embedding.iter().any(|x| !x.is_finite())
                {
                    return Err(ReferenceError::Numerical("routing embedding"));
                }
                let parent = parent_router
                    .route(
                        &embedding,
                        source.adapter_names(),
                        source.adapter_names().len(),
                    )
                    .map_err(|e| ReferenceError::Model(e.to_string()))?;
                let candidate = candidate_router
                    .route(
                        &embedding,
                        source.adapter_names(),
                        source.adapter_names().len(),
                    )
                    .map_err(|e| ReferenceError::Model(e.to_string()))?;
                mixture_inputs(adapters, &parent)?;
                mixture_inputs(adapters, &candidate)?;
                framed(
                    &mut execution,
                    &serde_json::to_vec(&(&prompt, &completion, &parent, &candidate))
                        .map_err(|e| invalid(e.to_string()))?,
                )?;
                for value in embedding {
                    execution.update(value.to_bits().to_be_bytes());
                }
                prepared.push((prompt, completion, parent, candidate));
            }
            let vocab = state.engine.config.vocab_size;
            let mut arm = Arm(state);
            let mut totals = [0.0f64; 3];
            for (prompt, completion, parent, candidate) in prepared {
                // A failing unsupported base arm yields no evidence, never a substitute.
                totals[0] += run_arm(&mut arm, adapters, &[], &prompt, &completion, vocab)?;
                totals[1] += run_arm(&mut arm, adapters, &parent, &prompt, &completion, vocab)?;
                totals[2] += run_arm(&mut arm, adapters, &candidate, &prompt, &completion, vocab)?;
            }
            let means = totals.map(|total| total / token_count as f64);
            if token_count == 0 || means.iter().any(|x| !x.is_finite()) {
                return Err(ReferenceError::Numerical("reference mean NLL"));
            }
            let mut evidence = ReferenceEvidence {
                contract_hash,
                parent_artifact_hash: source.artifact_hash().to_string(),
                parent_gate_hash: digest(source.gate_bytes()),
                candidate_gate_hash: digest(candidate.gate_bytes()),
                policy_hash: policy_hash(source)?,
                reference_hash,
                payload_hash,
                payload_summary,
                execution_hash: format!("{:x}", execution.finalize()),
                binding_hash: String::new(),
                token_count,
                base_mean_nll: means[0],
                parent_mean_nll: means[1],
                candidate_mean_nll: means[2],
                regression: means[2] - means[1],
            };
            if !evidence.regression.is_finite() {
                return Err(ReferenceError::Numerical("regression"));
            }
            evidence.binding_hash = evidence.compute_binding_hash()?;
            Ok(evidence)
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn reference_scoring_metal_blend_changes_nll_and_cleans_up() {
                let _gpu = crate::measurement::gpu_test_lock();
                let (cfg, weights) = crate::forward::metal_qwen35::reference_scoring_fixture();
                let mut state = MetalQwen35State::new(&weights, &cfg, 8)
                    .expect("reference witness requires a real Metal device");
                let layers = |sign: f32| {
                    let mut a = vec![0.0; cfg.full_q_dim()];
                    a[0] = 1.0;
                    let mut b = vec![0.0; cfg.hidden_size];
                    b[0] = sign;
                    vec![LoraLayerData {
                        layer_idx: 3,
                        module: "o_proj".into(),
                        a,
                        b,
                        rank: 1,
                        d_in: cfg.full_q_dim(),
                        d_out: cfg.hidden_size,
                    }]
                };
                let positive = layers(1.0);
                let negative = layers(-1.0);
                let descriptor = LoraDescriptor {
                    rank: 1,
                    alpha: 2.0,
                    target_modules: vec!["o_proj".into()],
                    dtype: "f32".into(),
                };
                let adapters = [
                    ReferenceAdapter {
                        name: "positive",
                        layers: &positive,
                        descriptor: &descriptor,
                        quarot_seed: None,
                    },
                    ReferenceAdapter {
                        name: "negative",
                        layers: &negative,
                        descriptor: &descriptor,
                        quarot_seed: None,
                    },
                ];
                let mixture = |positive| {
                    vec![
                        ("positive".to_string(), positive),
                        ("negative".to_string(), 1.0 - positive),
                    ]
                };
                assert!(mixture_inputs(&adapters, &[]).unwrap().is_empty());
                assert_eq!(
                    mixture_inputs(&adapters, &[("positive".into(), 1.0)])
                        .unwrap()
                        .len(),
                    1
                );
                let prompt = [4, 9, 13];
                let completion = [3, 3];
                let (base, parent, candidate, uniform) = {
                    let mut arm = Arm(&mut state);
                    let base = run_arm(
                        &mut arm,
                        &adapters,
                        &[],
                        &prompt,
                        &completion,
                        cfg.vocab_size,
                    )
                    .unwrap();
                    let parent = run_arm(
                        &mut arm,
                        &adapters,
                        &mixture(0.75),
                        &prompt,
                        &completion,
                        cfg.vocab_size,
                    )
                    .unwrap();
                    assert!(arm.0.has_lora_adapter());
                    let candidate = run_arm(
                        &mut arm,
                        &adapters,
                        &mixture(0.25),
                        &prompt,
                        &completion,
                        cfg.vocab_size,
                    )
                    .unwrap();
                    let uniform = run_arm(
                        &mut arm,
                        &adapters,
                        &mixture(0.5),
                        &prompt,
                        &completion,
                        cfg.vocab_size,
                    )
                    .unwrap();
                    (base, parent, candidate, uniform)
                };
                assert!(!state.has_lora_adapter());
                assert!(
                    (parent - candidate).abs() > 1e-3,
                    "distinct blends must change real NLL"
                );
                assert!(
                    (parent - uniform).abs() > 1e-3,
                    "uniform-weight mutation must fail"
                );
                assert!(
                    (candidate - uniform).abs() > 1e-3,
                    "uniform-weight mutation must fail"
                );
                assert!(
                    (uniform - base).abs() < 1e-3,
                    "opposite equal-scale payloads must cancel"
                );
                let direct_parent = {
                    let mut arm = Arm(&mut state);
                    // (0.75 - 0.25) * alpha/rank = 1, independent of mixture_inputs.
                    arm.0.load_lora_adapter(layers(1.0), 1.0, None).unwrap();
                    completion_nll(&mut arm, &prompt, &completion, cfg.vocab_size).unwrap()
                };
                assert!(
                    (direct_parent - parent).abs() < 1e-3,
                    "blend must include descriptor alpha/rank"
                );
                let base_again = {
                    let mut arm = Arm(&mut state);
                    // No run_arm reset: this prefill requires the previous guard's reset.
                    completion_nll(&mut arm, &prompt, &completion, cfg.vocab_size).unwrap()
                };
                assert!((base_again - base).abs() < 1e-6);
                {
                    let mut arm = Arm(&mut state);
                    let error = run_arm(
                        &mut arm,
                        &adapters,
                        &mixture(0.75),
                        &[4; 9],
                        &completion,
                        cfg.vocab_size,
                    );
                    assert!(matches!(error, Err(ReferenceError::Model(_))));
                    assert!(
                        arm.0.has_lora_adapter(),
                        "failure follows the real adapter load"
                    );
                }
                assert!(!state.has_lora_adapter());
                {
                    let mut arm = Arm(&mut state);
                    assert!(
                        (completion_nll(&mut arm, &prompt, &completion, cfg.vocab_size).unwrap()
                            - base)
                            .abs()
                            < 1e-6
                    );
                }
                eprintln!(
                    "[REFERENCE_METAL_BLEND] base={base} parent={parent} candidate={candidate} uniform={uniform} cleanup=success+error"
                );
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        struct Fake {
            rows: std::collections::VecDeque<Result<Vec<f32>, ReferenceError>>,
            calls: Vec<(Vec<u32>, Option<usize>)>,
        }
        impl TeacherForcer for Fake {
            fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>, ReferenceError> {
                self.calls.push((prompt.to_vec(), None));
                self.rows.pop_front().unwrap()
            }
            fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>, ReferenceError> {
                self.calls.push((vec![token], Some(position)));
                self.rows.pop_front().unwrap()
            }
        }
        #[test]
        fn reference_scoring_teacher_forcing_alignment_and_count() {
            let mut model = Fake {
                rows: [Ok(vec![0.0, 2.0, -1.0]), Ok(vec![1.0, -2.0, 3.0])].into(),
                calls: vec![],
            };
            let total = completion_nll(&mut model, &[0, 2, 0], &[1, 2], 3).unwrap();
            let expected = (1.0f64 + (-2.0f64).exp() + (-3.0f64).exp()).ln()
                + (1.0f64 + (-2.0f64).exp() + (-5.0f64).exp()).ln();
            assert!((total - expected).abs() < 1e-12);
            assert_eq!(model.calls, vec![(vec![0, 2, 0], None), (vec![1], Some(3))]);
            assert!(model.rows.is_empty());
            let mut single = Fake {
                rows: [Ok(vec![2.0, 0.0, 1.0])].into(),
                calls: vec![],
            };
            let nll = completion_nll(&mut single, &[1], &[2], 3).unwrap();
            assert!(
                (nll - (1.0 + (1.0f64 + (-1.0f64).exp() + (-2.0f64).exp()).ln())).abs() < 1e-12
            );
            assert_eq!(single.calls, vec![(vec![1], None)]);
        }
        #[test]
        fn reference_scoring_rejects_nonfinite_and_incomplete_vocab() {
            for logits in [
                vec![0.0, f32::NAN],
                vec![0.0, f32::INFINITY],
                vec![0.0, f32::NEG_INFINITY],
                vec![0.0],
            ] {
                assert!(full_vocab_nll(&logits, 0, 2).is_err());
            }
            assert!(full_vocab_nll(&[0.0, 1.0], 2, 2).is_err());
            assert!(
                full_vocab_nll(&[f32::MAX, -f32::MAX], 1, 2)
                    .unwrap()
                    .is_finite()
            );
        }
        #[test]
        fn reference_scoring_model_errors_and_empty_targets_refuse() {
            let mut model = Fake {
                rows: [
                    Ok(vec![0.0, 0.0]),
                    Err(ReferenceError::Model("step failed".into())),
                ]
                .into(),
                calls: vec![],
            };
            assert!(matches!(
                completion_nll(&mut model, &[0], &[1, 0], 2),
                Err(ReferenceError::Model(_))
            ));
            assert_eq!(model.calls.len(), 2);
            assert!(completion_nll(&mut model, &[0], &[], 2).is_err());
            assert!(completion_nll(&mut model, &[], &[0], 2).is_err());
            assert!(completion_nll(&mut model, &[2], &[0], 2).is_err());
            assert_eq!(model.calls.len(), 2);
            let mut failed = Fake {
                rows: [Err(ReferenceError::Model("prefill failed".into()))].into(),
                calls: vec![],
            };
            assert!(matches!(
                completion_nll(&mut failed, &[0], &[1], 2),
                Err(ReferenceError::Model(_))
            ));
            assert_eq!(failed.calls, vec![(vec![0], None)]);
        }
        #[test]
        fn reference_scoring_preserves_prompt_completion_token_boundary() {
            assert_eq!(completion_tokens(&[1, 2], &[1, 2, 3, 4]).unwrap(), &[3, 4]);
            assert!(completion_tokens(&[], &[1]).is_err());
            assert!(completion_tokens(&[1, 2], &[1, 3, 4]).is_err());
            assert!(completion_tokens(&[1, 2], &[1, 2]).is_err());
            assert!(completion_tokens(&[1, 2], &[1]).is_err());
        }
        #[test]
        fn reference_scoring_requires_explicit_bounds_and_content() {
            assert!(
                check_bounds(
                    ReferenceBounds {
                        max_examples: 0,
                        max_tokens: 3
                    },
                    1,
                    3
                )
                .is_err()
            );
            assert!(
                check_bounds(
                    ReferenceBounds {
                        max_examples: 1,
                        max_tokens: 0
                    },
                    1,
                    0
                )
                .is_err()
            );
            assert!(
                check_bounds(
                    ReferenceBounds {
                        max_examples: 1,
                        max_tokens: 3
                    },
                    2,
                    3
                )
                .is_err()
            );
            assert!(
                check_bounds(
                    ReferenceBounds {
                        max_examples: 1,
                        max_tokens: 3
                    },
                    1,
                    4
                )
                .is_err()
            );
            assert!(
                check_bounds(
                    ReferenceBounds {
                        max_examples: 1,
                        max_tokens: 3
                    },
                    1,
                    3
                )
                .is_ok()
            );
            let mut examples = vec![ReferenceExample {
                user_prompt: "ab".into(),
                completion: "c".into(),
            }];
            let original = reference_content_hash(&examples).unwrap();
            examples[0] = ReferenceExample {
                user_prompt: "a".into(),
                completion: "bc".into(),
            };
            assert_ne!(original, reference_content_hash(&examples).unwrap());
            assert!(reference_content_hash(&[]).is_err());
        }
    }
}

#[cfg(all(
    feature = "serve",
    feature = "f16",
    target_os = "macos",
    feature = "metal-gpu"
))]
pub use scoring::metal::{ReferenceAdapter, score_metal_reference};
