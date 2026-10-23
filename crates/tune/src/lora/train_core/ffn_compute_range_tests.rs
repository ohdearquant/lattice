use super::*;

/// Non-terminal layers must still run the FFN block at every position:
/// their output feeds the next layer's mixer (causal K/V, GDN state) at
/// every position, not just the completion positions.
#[test]
fn non_terminal_layer_uses_full_sequence() {
    let range = ffn_compute_range(false, 10, 6..9);
    assert_eq!(range, 0..10);
}

/// If the terminal-layer branch is dropped (or
/// inverted), the FFN block runs over the full prompt on the last layer
/// every step — a confirmed O(prompt_length) regression, since only the
/// completion positions' post-stack hidden state is ever read by the
/// head loop.
#[test]
fn terminal_layer_uses_completion_positions_only() {
    let range = ffn_compute_range(true, 10, 6..9);
    assert_eq!(range, 6..9);
    assert_ne!(range, 0..10, "must not degrade to the full-sequence range");
}

#[test]
fn terminal_layer_with_single_completion_position() {
    // A one-token completion (e.g. a 2-token pair) still only computes
    // FFN for that single position, not the whole prompt.
    let range = ffn_compute_range(true, 64, 62..63);
    assert_eq!(range, 62..63);
    assert_eq!(range.len(), 1);
}
