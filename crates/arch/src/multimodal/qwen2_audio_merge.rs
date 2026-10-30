//! Bounded single-clip embedding merge for the Transformers 4.45 Qwen2-Audio contract.
//!
//! This helper accepts one unpadded text sequence containing exactly one audio marker.
//! It neither tokenizes nor encodes audio, loads a checkpoint, or runs a decoder.

use lattice_inference_core::InferenceError;

// The 30-second Qwen2-Audio processor yields at most 750 projected rows.
const MAX_AUDIO_ROWS: usize = 750;
// This helper bounds prompt plus generation; this is not a model's context limit.
const MAX_CONTEXT: usize = 8192;
// The pinned Qwen2-Audio-7B text embedding width is 4096.
const MAX_WIDTH: usize = 4096;

/// **Unstable**: source of an expanded embedding row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSource {
    /// The row comes from this index of the original text embedding array.
    Text(usize),
    /// The row comes from this index of the projected audio embedding array.
    Audio(usize),
}

/// **Unstable**: explicit identity and resource bounds for one merge.
#[derive(Debug, Clone, Copy)]
pub struct MergeLimits {
    /// Audio marker token ID in the unexpanded text.
    pub audio_token_id: u32,
    /// HF legacy merge uses this ID for the audio feature positions.
    pub pad_token_id: u32,
    /// Vocabulary size for validating every supplied ID.
    pub vocab_size: usize,
    /// Shared embedding width of text and projected audio.
    pub hidden_size: usize,
    /// Total context capacity including requested generation.
    pub context_limit: usize,
    /// Number of positions reserved for subsequent generation.
    pub generation_budget: usize,
}

/// **Unstable**: immutable plan for one unpadded, unexpanded single-clip prompt.
#[derive(Debug, Clone)]
pub struct SingleAudioPlan {
    // Preserve the original identity so splice cannot silently accept another prompt.
    input_ids: Vec<u32>,
    final_ids: Vec<u32>,
    sources: Vec<RowSource>,
    audio_rows: usize,
    hidden_size: usize,
}

fn invalid(message: &str) -> InferenceError {
    InferenceError::InvalidInput(format!("qwen2-audio merge: {message}"))
}

fn validate_input(input_ids: &[u32], attention_mask: &[u8]) -> Result<(), InferenceError> {
    if input_ids.is_empty() || input_ids.len() != attention_mask.len() {
        return Err(invalid("empty input or attention mask length mismatch"));
    }
    if attention_mask.iter().any(|&value| value != 1) {
        return Err(invalid("only one unpadded sequence is supported"));
    }
    Ok(())
}

fn validate_limits(audio_rows: usize, limits: &MergeLimits) -> Result<(), InferenceError> {
    if !(1..=MAX_AUDIO_ROWS).contains(&audio_rows)
        || !(1..=MAX_WIDTH).contains(&limits.hidden_size)
        || !(1..=MAX_CONTEXT).contains(&limits.context_limit)
        || limits.vocab_size == 0
    {
        return Err(invalid("audio, width, vocabulary or context bounds"));
    }
    Ok(())
}

fn validate_token_identity(input_ids: &[u32], limits: &MergeLimits) -> Result<(), InferenceError> {
    if limits.audio_token_id == limits.pad_token_id
        || limits.audio_token_id as usize >= limits.vocab_size
        || limits.pad_token_id as usize >= limits.vocab_size
        || input_ids.iter().any(|&id| id as usize >= limits.vocab_size)
        || input_ids.contains(&limits.pad_token_id)
    {
        return Err(invalid("invalid token identity or reserved pad token"));
    }
    Ok(())
}

fn find_marker(input_ids: &[u32], audio_token_id: u32) -> Result<usize, InferenceError> {
    let mut marker = None;
    for (index, &id) in input_ids.iter().enumerate() {
        if id == audio_token_id && marker.replace(index).is_some() {
            return Err(invalid("expected exactly one audio marker"));
        }
    }
    marker.ok_or_else(|| invalid("expected exactly one audio marker"))
}

fn checked_expanded_len(
    input_len: usize,
    audio_rows: usize,
    limits: &MergeLimits,
) -> Result<usize, InferenceError> {
    let expanded_len = input_len
        .checked_sub(1)
        .and_then(|n| n.checked_add(audio_rows))
        .ok_or_else(|| invalid("expanded length overflow"))?;
    if expanded_len
        .checked_add(limits.generation_budget)
        .is_none_or(|n| n > limits.context_limit)
    {
        return Err(invalid("expanded prompt and generation exceed context"));
    }
    Ok(expanded_len)
}

impl SingleAudioPlan {
    /// Build a plan after validating identity, marker count and context bounds.
    ///
    /// `attention_mask` must contain one for every input token; padded and batched
    /// sequences, empty audio, and repeated or absent markers are not supported.
    ///
    /// # Errors
    /// Returns an error for an unsupported layout, invalid IDs, or exceeded bounds.
    pub fn new(
        input_ids: &[u32],
        attention_mask: &[u8],
        audio_rows: usize,
        limits: MergeLimits,
    ) -> Result<Self, InferenceError> {
        validate_input(input_ids, attention_mask)?;
        validate_limits(audio_rows, &limits)?;
        validate_token_identity(input_ids, &limits)?;
        let marker = find_marker(input_ids, limits.audio_token_id)?;
        let expanded_len = checked_expanded_len(input_ids.len(), audio_rows, &limits)?;
        let mut final_ids = Vec::with_capacity(expanded_len);
        let mut sources = Vec::with_capacity(expanded_len);
        for (index, &id) in input_ids.iter().enumerate() {
            if index == marker {
                for row in 0..audio_rows {
                    final_ids.push(limits.pad_token_id);
                    sources.push(RowSource::Audio(row));
                }
            } else {
                final_ids.push(id);
                sources.push(RowSource::Text(index));
            }
        }
        Ok(Self {
            input_ids: input_ids.to_vec(),
            final_ids,
            sources,
            audio_rows,
            hidden_size: limits.hidden_size,
        })
    }

    /// Legacy HF output IDs: audio rows have the pad ID, not repeated marker IDs.
    pub fn final_input_ids(&self) -> &[u32] {
        &self.final_ids
    }

    /// Ordered text/audio source rows; all expanded positions are attended.
    pub fn sources(&self) -> &[RowSource] {
        &self.sources
    }

    /// Expanded sequence length. Position IDs are `0..len` for this unpadded case.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the plan has no rows. Valid plans are nonempty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Write the merged embeddings to caller-owned storage after all validation.
    ///
    /// # Errors
    /// Returns an error if IDs, lengths, dimensions or values do not match the plan.
    /// On error `output` is not modified.
    pub fn splice(
        &self,
        input_ids: &[u32],
        text_embeddings: &[f32],
        audio_embeddings: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        if input_ids != self.input_ids.as_slice()
            || self.input_ids.len().checked_mul(self.hidden_size) != Some(text_embeddings.len())
            || self.audio_rows.checked_mul(self.hidden_size) != Some(audio_embeddings.len())
            || self.len().checked_mul(self.hidden_size) != Some(output.len())
        {
            return Err(invalid("embedding shape or input identity mismatch"));
        }
        if text_embeddings
            .iter()
            .chain(audio_embeddings)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("embedding contains a non-finite value"));
        }
        for (dest, source) in output.chunks_exact_mut(self.hidden_size).zip(&self.sources) {
            let (values, index) = match *source {
                RowSource::Text(index) => (text_embeddings, index),
                RowSource::Audio(index) => (audio_embeddings, index),
            };
            let start = index * self.hidden_size;
            dest.copy_from_slice(&values[start..start + self.hidden_size]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
