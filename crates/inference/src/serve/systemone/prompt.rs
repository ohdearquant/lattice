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
mod tests {
    use super::*;
    use crate::tokenizer::TokenizedInput;

    /// A tokenizer that maps whole strings to fixed ids, so assembly can be
    /// checked without a real vocabulary.
    struct ScriptedTokenizer;

    impl Tokenizer for ScriptedTokenizer {
        fn tokenize(&self, text: &str) -> TokenizedInput {
            let ids: Vec<u32> = text.bytes().map(u32::from).collect();
            let length = ids.len();
            TokenizedInput {
                input_ids: ids,
                attention_mask: vec![1; length],
                token_type_ids: vec![0; length],
                real_length: length,
                pre_truncation_len: length,
            }
        }

        fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
            texts.iter().map(|text| self.tokenize(text)).collect()
        }

        fn vocab_size(&self) -> usize {
            256
        }

        fn max_seq_len(&self) -> usize {
            usize::MAX
        }
    }

    /// A byte-level tokenizer, except that a space immediately followed by a
    /// newline is emitted as one id. The reference encodes the whole joined
    /// question string in a single call on the narrow path, so a boundary that
    /// falls inside that string is visible here; an assembly that encodes the
    /// head, each option line and the tail separately cannot produce the merged
    /// id, because the space and the newline land in different calls.
    struct MergingTokenizer;

    /// The id [`MergingTokenizer`] emits for a space followed by a newline; it
    /// sits outside the byte range so it cannot collide with a raw byte.
    const MERGED_SPACE_NEWLINE: u32 = 2000;

    impl Tokenizer for MergingTokenizer {
        fn tokenize(&self, text: &str) -> TokenizedInput {
            let bytes = text.as_bytes();
            let mut ids = Vec::with_capacity(bytes.len());
            let mut index = 0;
            while index < bytes.len() {
                if bytes[index] == b' ' && bytes.get(index + 1) == Some(&b'\n') {
                    ids.push(MERGED_SPACE_NEWLINE);
                    index += 2;
                } else {
                    ids.push(u32::from(bytes[index]));
                    index += 1;
                }
            }
            let length = ids.len();
            TokenizedInput {
                input_ids: ids,
                attention_mask: vec![1; length],
                token_type_ids: vec![0; length],
                real_length: length,
                pre_truncation_len: length,
            }
        }

        fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
            texts.iter().map(|text| self.tokenize(text)).collect()
        }

        fn vocab_size(&self) -> usize {
            MERGED_SPACE_NEWLINE as usize + 1
        }

        fn max_seq_len(&self) -> usize {
            usize::MAX
        }
    }

    fn config(max_state_tokens: usize) -> DeciderConfig {
        DeciderConfig {
            version: "0.8b-v1".to_string(),
            temperature: 1.03,
            max_state_tokens,
            max_options: 255,
            isolated_levels: true,
        }
    }

    fn table(entries: usize) -> DecisionLabelTable {
        DecisionLabelTable {
            labels: (0..entries).map(|index| format!("L{index}")).collect(),
            ids: (0..entries as u32).map(|index| 1000 + index).collect(),
            open_ids: vec![7, 8],
        }
    }

    fn rows(question: &str, options: &[&str]) -> Vec<PlannedRow> {
        vec![PlannedRow {
            question: question.to_string(),
            options: options.iter().map(ToString::to_string).collect(),
            id: "q".to_string(),
        }]
    }

    #[test]
    fn narrow_row_is_one_encode() {
        let prompt = assemble_prompt(
            &ScriptedTokenizer,
            "s",
            &rows("?", &["a", "b"]),
            &config(32768),
            &table(10),
        )
        .expect("assembles");
        let row = &prompt.rows[0];
        let expected: Vec<u32> = b"Context:\ns\n\nQuestion: ?\nOptions:\n(A) a\n(B) b\nAnswer: ("
            .iter()
            .map(|byte| u32::from(*byte))
            .collect();
        assert_eq!(row.ids, expected);
        assert_eq!(row.slot, row.ids.len() - 1);
        assert_eq!(row.nopts, 2);
    }

    #[test]
    fn narrow_row_merges_across_the_option_boundary() {
        // The first option ends in a space and the next option line begins with
        // a newline, so the two characters meet inside the string the reference
        // joins and encodes once. A per-piece assembly encodes them separately
        // and cannot emit the merged id.
        let state = "s";
        let piece = "\n\nQuestion: ?\nOptions:\n(A) a \n(B) b\nAnswer: (";
        let prompt = assemble_prompt(
            &MergingTokenizer,
            state,
            &rows("?", &["a ", "b"]),
            &config(32768),
            &table(10),
        )
        .expect("assembles");
        let row = &prompt.rows[0];
        // The state and the question piece are encoded in separate calls, as the
        // reference does; only the question piece is joined before encoding.
        let mut expected = MergingTokenizer
            .tokenize(&format!("Context:\n{state}"))
            .input_ids;
        expected.extend(MergingTokenizer.tokenize(piece).input_ids);
        assert_eq!(row.ids, expected);
        assert!(
            row.ids.contains(&MERGED_SPACE_NEWLINE),
            "the merged space-and-newline id must appear in the row"
        );
        assert_eq!(
            row.ids
                .iter()
                .filter(|id| **id == MERGED_SPACE_NEWLINE)
                .count(),
            1
        );
    }

    #[test]
    fn wide_row_splices_label_ids() {
        let options: Vec<String> = (0..11).map(|index| format!("o{index}")).collect();
        let options: Vec<&str> = options.iter().map(String::as_str).collect();
        let prompt = assemble_prompt(
            &ScriptedTokenizer,
            "s",
            &rows("?", &options),
            &config(32768),
            &table(255),
        )
        .expect("assembles");
        let row = &prompt.rows[0];
        // The eleventh label id must appear in the ids, spliced in by hand.
        assert!(row.ids.contains(&1010));
        assert_eq!(
            row.ids.iter().filter(|id| **id == 1000).count(),
            1,
            "label 0 appears once"
        );
    }

    #[test]
    fn state_is_sliced_and_reported() {
        let max_state_tokens = 5usize;
        let state = "w ".repeat(10);
        let prompt = assemble_prompt(
            &ScriptedTokenizer,
            &state,
            &rows("?", &["a", "b"]),
            &config(max_state_tokens),
            &table(10),
        )
        .expect("assembles");
        assert!(prompt.truncated);
        let state_full = ScriptedTokenizer
            .tokenize(&format!("Context:\n{state}"))
            .input_ids;
        assert_eq!(prompt.state_tokens, state_full.len());
        // The reference slices only the state, then appends the question's own
        // encode, so the row is the state prefix followed by the whole question
        // piece for that row.
        let question = ScriptedTokenizer
            .tokenize("\n\nQuestion: ?\nOptions:\n(A) a\n(B) b\nAnswer: (")
            .input_ids;
        let row = &prompt.rows[0].ids;
        assert_eq!(&row[..max_state_tokens], &state_full[..max_state_tokens]);
        assert_eq!(&row[max_state_tokens..], question.as_slice());
    }

    #[test]
    fn unique_tokens_counts_the_prefix_once() {
        let prompt = assemble_prompt(
            &ScriptedTokenizer,
            "s",
            &rows("?", &["a", "b"]),
            &config(32768),
            &table(10),
        )
        .expect("assembles");
        assert_eq!(prompt.unique_tokens(), prompt.rows[0].ids.len());
    }
}
