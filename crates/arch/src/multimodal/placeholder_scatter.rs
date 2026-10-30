//! Placement of encoder output rows at placeholder-token positions.
//!
//! This is the operation Hugging Face multimodal models perform with
//! `inputs_embeds.masked_scatter(mask, features)`: the i-th encoder row
//! replaces the embedding row of the i-th placeholder token.

use lattice_inference_core::InferenceError;

fn invalid(message: String) -> InferenceError {
    InferenceError::InvalidInput(format!("placeholder scatter: {message}"))
}

/// **Unstable**: positions of `placeholder_id` in `ids`, in order.
pub fn placeholder_slots(ids: &[u32], placeholder_id: u32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(pos, &id)| (id == placeholder_id).then_some(pos))
        .collect()
}

/// **Unstable**: overwrite row `slots[i]` of `embeds` with row `i` of `rows`.
///
/// `embeds` and `rows` are row-major with `hidden` values per row.
///
/// # Errors
/// Returns an error when `hidden` is zero, `embeds` is not a whole number of rows,
/// `rows` does not hold exactly one row per slot, or the slots are not strictly
/// increasing positions inside `embeds`. On error `embeds` is not modified.
pub fn scatter_rows(
    slots: &[usize],
    rows: &[f32],
    hidden: usize,
    embeds: &mut [f32],
) -> Result<(), InferenceError> {
    if hidden == 0 || !embeds.len().is_multiple_of(hidden) {
        return Err(invalid(format!(
            "{} embedding values are not whole rows of width {hidden}",
            embeds.len()
        )));
    }
    if slots.len().checked_mul(hidden) != Some(rows.len()) {
        return Err(invalid(format!(
            "{} slots need {} row values, got {}",
            slots.len(),
            slots.len().saturating_mul(hidden),
            rows.len()
        )));
    }
    let seq_len = embeds.len() / hidden;
    if slots.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid("slots are not strictly increasing".to_string()));
    }
    if let Some(&last) = slots.last()
        && last >= seq_len
    {
        return Err(invalid(format!(
            "slot {last} is outside a sequence of {seq_len} rows"
        )));
    }
    for (row, &slot) in rows.chunks_exact(hidden).zip(slots) {
        embeds[slot * hidden..(slot + 1) * hidden].copy_from_slice(row);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
