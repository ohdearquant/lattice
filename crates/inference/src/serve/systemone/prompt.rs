//! Token-level prompt assembly for the system-one state-first layout.
//!
//! A byte-exact port of the reference's `prompt.build` on the `state_first`
//! layout that `serve._prepare_s1` uses: the state is encoded once and sliced to
//! the checkpoint's truncation length, then each row appends one question. A row
//! with at most ten options is encoded as one string (the `(A) .. (J)`
//! rendering); a wider row is built from ids so every label is exactly one
//! token, splicing the label ids around per-option encodes.
//!
//! The reference's `encode` applies no special tokens, which corresponds to
//! [`Tokenizer::tokenize`] on a tokenizer the port has configured without
//! start/end markers: the decider checkpoint's `post_processor` is a `ByteLevel`
//! one, not a template, so no marker is added. The port additionally requires
//! the tokenizer's window to be wide enough that a single encode never
//! truncates; the reference slices the state itself, and lattice's own
//! truncation would otherwise fire on the assembled row.

use crate::option_scoring::DecisionLabelTable;
use crate::tokenizer::Tokenizer;

use super::SystemOneError;
use super::config::DeciderConfig;
use super::request::{LETTER_LABELS, PlannedRow};

/// The fixed letters of the narrow rendering, in order.
const LETTERS: &[u8] = b"ABCDEFGHIJ";

/// One assembled row: its token ids and the slot the answer is read at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowPrompt {
    /// The row's token ids.
    pub ids: Vec<u32>,
    /// The index of the last id, where the answer letter is read.
    pub slot: usize,
    /// The row's option count.
    pub nopts: usize,
}

/// The assembled prompt for one request: one row per planned row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledPrompt {
    /// The rows, in planned order.
    pub rows: Vec<RowPrompt>,
    /// The token count of `"Context:\n" + state` before the state was sliced.
    pub state_tokens: usize,
    /// Whether the state was sliced to the checkpoint's truncation length.
    pub truncated: bool,
}

impl AssembledPrompt {
    /// The input token count the reference reports: the rows' longest common
    /// prefix counted once.
    pub fn unique_tokens(&self) -> usize {
        if self.rows.len() < 2 {
            return self.rows.iter().map(|row| row.ids.len()).sum();
        }
        let shortest = self.rows.iter().map(|row| row.ids.len()).min().unwrap_or(0);
        let first = &self.rows[0].ids;
        let mut common = 0usize;
        while common < shortest && self.rows.iter().all(|row| row.ids[common] == first[common]) {
            common += 1;
        }
        common
            + self
                .rows
                .iter()
                .map(|row| row.ids.len() - common)
                .sum::<usize>()
    }
}

/// Encode `text` with no truncation. Refuses a tokenizer whose window is too
/// small, because the reference applies its own slice and a lattice-side
/// truncation would silently change the row.
///
/// A one-element batch is used rather than [`Tokenizer::tokenize`] so the result
/// is padded only to its own length, not to `max_seq_len` (which for a
/// long-context checkpoint is tens of thousands of ids per call).
pub fn encode(tokenizer: &dyn Tokenizer, text: &str) -> Result<Vec<u32>, SystemOneError> {
    let mut batch = tokenizer.tokenize_batch(&[text]);
    let input = batch
        .pop()
        .ok_or_else(|| SystemOneError::Refused("the tokenizer returned no ids".to_string()))?;
    if input.pre_truncation_len != input.real_length {
        return Err(SystemOneError::Refused(format!(
            "the tokenizer window truncated a {} token encode at {} tokens",
            input.pre_truncation_len, input.real_length
        )));
    }
    input
        .input_ids
        .get(..input.real_length)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| SystemOneError::Refused("the tokenizer returned no ids".to_string()))
}

/// Assemble the prompt for one request.
pub fn assemble_prompt(
    tokenizer: &dyn Tokenizer,
    state: &str,
    rows: &[PlannedRow],
    config: &DeciderConfig,
    labels: &DecisionLabelTable,
) -> Result<AssembledPrompt, SystemOneError> {
    let state_full = encode(tokenizer, &format!("Context:\n{state}"))?;
    let state_tokens = state_full.len();
    let truncated = state_tokens > config.max_state_tokens;
    let keep = state_tokens.min(config.max_state_tokens);
    let prefix = state_full.get(..keep).unwrap_or(&state_full);

    let mut assembled = Vec::with_capacity(rows.len());
    for row in rows {
        // Each row is built from the state prefix on its own (the reference
        // builds one `Example` per row), so the prefix is copied, not shared.
        let mut ids: Vec<u32> = prefix.to_vec();
        let question = format!("\n\nQuestion: {}\nOptions:", row.question);
        let tail = "\nAnswer: (";
        if row.options.len() <= LETTER_LABELS {
            let mut piece = String::new();
            piece.push_str(&question);
            for (index, option) in row.options.iter().enumerate() {
                let letter = char::from(*LETTERS.get(index).unwrap_or(&b'?'));
                piece.push_str(&format!("\n({letter}) {option}"));
            }
            piece.push_str(tail);
            ids.extend(encode(tokenizer, &piece)?);
        } else {
            ids.extend(encode(tokenizer, &question)?);
            for (index, option) in row.options.iter().enumerate() {
                ids.extend_from_slice(&labels.open_ids);
                let Some(&label_id) = labels.ids.get(index) else {
                    return Err(SystemOneError::Refused(format!(
                        "the label table has {} entries, option {index} needs one",
                        labels.ids.len()
                    )));
                };
                ids.push(label_id);
                ids.extend(encode(tokenizer, &format!(") {option}"))?);
            }
            ids.extend(encode(tokenizer, tail)?);
        }
        assembled.push(RowPrompt {
            slot: ids.len().saturating_sub(1),
            nopts: row.options.len(),
            ids,
        });
    }

    Ok(AssembledPrompt {
        rows: assembled,
        state_tokens,
        truncated,
    })
}

#[cfg(test)]
mod tests;
